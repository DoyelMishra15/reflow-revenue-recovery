"""
Decides what to do with a failed payment: retry now, delay and retry later,
suggest an alternate payment method, or give up and hand off to a human /
notify the customer.

Design note: the ML model only outputs a probability. It never gets to make
an unbounded decision on its own - every model output passes through hard
rule-based gates before it can trigger an action. This is deliberate: a
classifier trained on 6k synthetic rows should not be trusted to decide,
by itself, when it's OK to auto-retry someone's card. The gates are the
actual safety layer; the model just informs which side of a gate we land on.
"""
from dataclasses import dataclass, field
from enum import Enum

from app.ml.classifier import predict_retry_success_proba
from app.ml.features import to_feature_dict

MAX_AUTO_RETRIES = 3

# Below this predicted success probability, we won't auto-retry. This is
# deliberately low. A retry only costs COST_PER_WASTED_RETRY (~₹8 of
# operational friction, see evaluation/evaluate.py) against a txn amount
# that's almost always orders of magnitude bigger - so from a pure
# expected-value view, it's worth attempting almost anything with a
# non-trivial chance of clearing. The floor exists to cut the genuinely
# low-probability tail (the transactions the model is confident won't
# clear), not to hold out for high-confidence wins at the cost of revenue.
# 0.20 is where that tail starts: below it, retries succeed well under a
# third of the time and mostly just add issuer-side friction for nothing.
# See backend/evaluation/metrics_output.json for the measured trade-off at
# this threshold vs. a pure rule-based baseline.
MIN_PROBA_TO_RETRY = 0.20
COOLDOWN_MINUTES_BY_REASON = {
    "bank_server_down": 30,
    "network_timeout": 15,
}
DEFAULT_COOLDOWN_MINUTES = 5
NO_AUTO_RETRY_REASONS = {"card_expired", "risk_block"}
LARGE_TXN_THRESHOLD = 100_000  # paise-agnostic; this is rupees in our mock data

# root-cause bucket per failure_reason - this is the "diagnosis" step. Kept as a
# static lookup rather than a model because these categories are definitional
# (issuer_declined just *is* an issuer-side thing), not something worth spending
# a classifier on. The model's job downstream is scoring recoverability, not
# labeling the cause.
DIAGNOSIS_BY_REASON = {
    "insufficient_funds": "customer-side, funds",
    "card_expired": "customer-side, instrument invalid",
    "invalid_otp": "customer-side, auth friction",
    "issuer_declined": "issuer-side, policy decline",
    "bank_server_down": "bank-side, infra outage",
    "network_timeout": "network/infra, transient",
    "gateway_error": "gateway-side, transient",
    "risk_block": "fraud/risk signal",
}


class Action(str, Enum):
    RETRY_NOW = "retry_now"
    DELAY_RETRY = "delay_retry"
    SUGGEST_ALT_METHOD = "suggest_alt_method"
    ESCALATE_HUMAN = "escalate_human"
    STOP_NO_RETRY = "stop_no_retry"


def diagnose(txn: dict) -> str:
    return DIAGNOSIS_BY_REASON.get(txn["failure_reason"], "unclassified")


@dataclass
class Decision:
    action: Action
    reason_code: str
    explanation: str
    diagnosis: str = "unclassified"
    predicted_success_proba: float | None = None
    cooldown_minutes: int | None = None
    gates_triggered: list[str] = field(default_factory=list)


def decide(txn: dict) -> Decision:
    """txn is a dict with the raw transaction fields (see app/schemas.py)."""
    d = _decide_action(txn)
    d.diagnosis = diagnose(txn)
    return d


def _decide_action(txn: dict) -> Decision:
    gates_triggered = []

    # --- hard safety gates (checked before we even look at the model) ---
    if txn.get("is_flagged_risk"):
        gates_triggered.append("risk_flag")
        return Decision(
            action=Action.ESCALATE_HUMAN,
            reason_code="risk_flagged",
            explanation="Transaction flagged by risk checks - auto-retry disabled, routed to fraud review queue.",
            gates_triggered=gates_triggered,
        )

    if txn["failure_reason"] in NO_AUTO_RETRY_REASONS:
        gates_triggered.append("non_recoverable_reason")
        return Decision(
            action=Action.STOP_NO_RETRY,
            reason_code=txn["failure_reason"],
            explanation=f"'{txn['failure_reason']}' is not recoverable by retrying (needs new card/verification). "
            "Notifying customer instead of burning a retry attempt.",
            gates_triggered=gates_triggered,
        )

    if txn["attempt_no"] > MAX_AUTO_RETRIES:
        gates_triggered.append("max_retries_exceeded")
        return Decision(
            action=Action.ESCALATE_HUMAN,
            reason_code="max_retries_exceeded",
            explanation=f"Already attempted {txn['attempt_no']} times, past the cap of {MAX_AUTO_RETRIES}. "
            "Stopping automated retries to avoid annoying the customer / tripping issuer fraud rules.",
            gates_triggered=gates_triggered,
        )

    if txn["amount"] > LARGE_TXN_THRESHOLD and txn["attempt_no"] > 1:
        gates_triggered.append("large_txn_repeat_attempt")
        return Decision(
            action=Action.ESCALATE_HUMAN,
            reason_code="high_value_repeat_failure",
            explanation=f"Amount ₹{txn['amount']:.0f} exceeds ₹{LARGE_TXN_THRESHOLD:,} and this is attempt "
            f"#{txn['attempt_no']}. Routing to manual review rather than auto-retrying a large sum blind.",
            gates_triggered=gates_triggered,
        )

    # --- past the gates, ask the model for its read on this one ---
    proba = predict_retry_success_proba(to_feature_dict(txn))

    cooldown = COOLDOWN_MINUTES_BY_REASON.get(txn["failure_reason"], DEFAULT_COOLDOWN_MINUTES)
    too_soon = txn["minutes_since_last_failure"] < cooldown

    if proba < MIN_PROBA_TO_RETRY:
        if txn["payment_method"] != "upi" and txn["failure_reason"] in ("issuer_declined", "insufficient_funds"):
            return Decision(
                action=Action.SUGGEST_ALT_METHOD,
                reason_code="low_success_proba_alt_available",
                explanation=(
                    f"Model gives this a {proba:.0%} retry-success chance on {txn['payment_method']} - low. "
                    "Issuer-side declines often clear on a different rail, so suggesting UPI instead of retrying blind."
                ),
                predicted_success_proba=proba,
            )
        return Decision(
            action=Action.STOP_NO_RETRY,
            reason_code="low_predicted_success",
            explanation=f"Model gives this only a {proba:.0%} chance of succeeding on retry - below the "
            f"{MIN_PROBA_TO_RETRY:.0%} threshold. Not worth the customer friction of another attempt.",
            predicted_success_proba=proba,
        )

    if too_soon:
        return Decision(
            action=Action.DELAY_RETRY,
            reason_code="cooldown_active",
            explanation=f"Predicted success is {proba:.0%} (good), but only {txn['minutes_since_last_failure']}min "
            f"have passed and '{txn['failure_reason']}' usually needs {cooldown}min to clear "
            "(bank-side issue, not the customer's). Scheduling a delayed retry instead of hammering it now.",
            predicted_success_proba=proba,
            cooldown_minutes=cooldown - txn["minutes_since_last_failure"],
        )

    return Decision(
        action=Action.RETRY_NOW,
        reason_code="high_predicted_success",
        explanation=f"Model gives this a {proba:.0%} chance of succeeding and cooldown has cleared. Retrying now.",
        predicted_success_proba=proba,
    )
