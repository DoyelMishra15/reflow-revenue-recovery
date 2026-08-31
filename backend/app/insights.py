"""
Everything in this module is additive analysis layered on top of the
existing retry_engine / ground_truth / evaluate machinery - it never
redefines "what should happen to a payment" (that's still retry_engine.decide,
untouched) and never fabricates numbers that aren't derived from real
transaction/retry/audit rows or the same hidden recoverability function the
mock gateway itself uses. Nothing here mutates the database; every function
takes data in and returns computed data out.
"""
import random
from collections import defaultdict
from dataclasses import dataclass

from app.ground_truth import true_recovery_probability
from app.ml.classifier import predict_retry_success_proba
from app.ml.features import to_feature_dict
from app.retry_engine import (
    Action,
    COOLDOWN_MINUTES_BY_REASON,
    DEFAULT_COOLDOWN_MINUTES,
    DIAGNOSIS_BY_REASON,
    LARGE_TXN_THRESHOLD,
    MAX_AUTO_RETRIES,
    NO_AUTO_RETRY_REASONS,
    diagnose,
)

COST_PER_WASTED_RETRY = 8.0  # mirrors evaluation/evaluate.py

CUSTOMER_SIDE_REASONS = {"insufficient_funds", "card_expired", "invalid_otp"}
SYSTEMIC_REASONS = {"issuer_declined", "bank_server_down", "network_timeout", "gateway_error"}
# risk_block is neither - it's a fraud signal, handled separately in classification

OPEN_STATUSES = ("failed", "delayed", "alt_method_suggested")


def _txn_to_dict(txn) -> dict:
    return {
        "txn_id": txn.txn_id,
        "amount": txn.amount,
        "payment_method": txn.payment_method,
        "bank": txn.bank,
        "failure_reason": txn.failure_reason,
        "attempt_no": txn.attempt_no,
        "hour_of_day": txn.hour_of_day,
        "is_weekend": txn.is_weekend,
        "customer_past_success_rate": txn.customer_past_success_rate,
        "minutes_since_last_failure": txn.minutes_since_last_failure,
        "is_flagged_risk": txn.is_flagged_risk,
        "status": txn.status,
    }


def _classify(failure_reason: str) -> str:
    if failure_reason in CUSTOMER_SIDE_REASONS:
        return "customer"
    if failure_reason == "risk_block":
        return "fraud"
    return "systemic"


# ---------------------------------------------------------------------------
# 1. Systemic failure clustering / incident detection
# ---------------------------------------------------------------------------

MIN_CLUSTER_SIZE = 3          # below this, it's just noise, not a pattern
MIN_CLUSTER_SHARE = 0.12      # a cluster must be >=12% of *systemic* failure volume to flag

RECOMMENDED_ACTION_BY_REASON = {
    "bank_server_down": "Suppress retries on affected bank; queue for delayed retry once outage clears.",
    "network_timeout": "Suppress immediate retries; reroute eligible payments to a stable rail (UPI/netbanking).",
    "issuer_declined": "Reroute eligible customers to an alternate payment method rather than retrying the same card.",
    "gateway_error": "Suppress retries at the gateway; failover to backup processor if available.",
    "risk_block": "Escalate to fraud review; do not auto-retry.",
    "insufficient_funds": "Hold and notify customer - not a systemic issue, no action needed at the bank/issuer level.",
    "card_expired": "Notify customer to update instrument - not systemic.",
    "invalid_otp": "Notify customer to retry auth - not systemic.",
}


def detect_incidents(transactions: list) -> list[dict]:
    """
    Groups failed transactions by (bank, failure_reason) and by
    (payment_method, failure_reason) and flags any group that's large
    enough, and concentrated enough, to look like a systemic degradation
    rather than independent customer-side failures.
    """
    failed = [t for t in transactions if t.status in OPEN_STATUSES + ("given_up", "escalated")]
    systemic_failed = [t for t in failed if _classify(t.failure_reason) == "systemic"]
    # denominator is systemic-failure volume specifically, not all failures -
    # customer-side failures (insufficient_funds, card_expired, ...) are
    # independent per-customer events and would just dilute a real cluster
    # if lumped into the same share calculation
    total = len(systemic_failed) or 1

    groups: dict[tuple, list] = defaultdict(list)
    for t in systemic_failed:
        groups[("bank", t.bank, t.failure_reason)].append(t)
        groups[("method", t.payment_method, t.failure_reason)].append(t)

    incidents = []
    seen_txn_sets = set()
    for (dim, key, reason), members in groups.items():
        share = len(members) / total
        if len(members) < MIN_CLUSTER_SIZE or share < MIN_CLUSTER_SHARE:
            continue
        txn_ids = tuple(sorted(m.txn_id for m in members))
        if txn_ids in seen_txn_sets:
            continue
        seen_txn_sets.add(txn_ids)

        amount_at_risk = sum(m.amount for m in members)
        confidence = min(0.97, 0.4 + share + (0.05 * len(members)))
        baseline_share = 1.0 / max(len({t.failure_reason for t in failed}), 1)
        spike_multiplier = round(share / baseline_share, 1) if baseline_share else 1.0

        incidents.append(
            {
                "id": f"incident_{dim}_{key}_{reason}",
                "dimension": dim,  # "bank" or "method"
                "affected": key,
                "failure_reason": reason,
                "diagnosis": DIAGNOSIS_BY_REASON.get(reason, "unclassified"),
                "affected_count": len(members),
                "affected_share_pct": round(share * 100, 1),
                "spike_vs_baseline": f"{spike_multiplier}x",
                "amount_at_risk": round(amount_at_risk, 2),
                "confidence": round(confidence, 2),
                "recommended_action": RECOMMENDED_ACTION_BY_REASON.get(
                    reason, "Route to manual review."
                ),
                "affected_txn_ids": list(txn_ids),
            }
        )

    incidents.sort(key=lambda i: i["amount_at_risk"], reverse=True)
    return incidents


def build_action_plan(incident: dict) -> list[dict]:
    """Concrete, orderable steps for an incident - what 'Execute plan' simulates."""
    reason = incident["failure_reason"]
    steps = []
    if reason in ("bank_server_down", "gateway_error", "network_timeout"):
        steps.append({"action": "suppress_retries", "detail": f"Pause auto-retry for {incident['affected_count']} affected transactions to avoid hammering a degraded endpoint."})
        steps.append({"action": "delay_retry", "detail": f"Re-queue after cooldown ({COOLDOWN_MINUTES_BY_REASON.get(reason, DEFAULT_COOLDOWN_MINUTES)} min) once the endpoint recovers."})
    if reason == "issuer_declined":
        steps.append({"action": "reroute_eligible", "detail": "Suggest UPI/netbanking to customers on non-UPI methods instead of retrying the declined card."})
    if reason == "risk_block":
        steps.append({"action": "escalate", "detail": "Route all affected transactions to fraud review; no auto-retry."})
    steps.append({"action": "notify_customers", "detail": "Send a status update to affected customers so they don't retry manually and double-charge."})
    return steps


# ---------------------------------------------------------------------------
# 2. Reflow vs. blind retry counterfactual
# ---------------------------------------------------------------------------

def _seeded_rng(txn_id: str) -> random.Random:
    """Deterministic per-transaction RNG so the counterfactual is stable across calls."""
    return random.Random(f"blind::{txn_id}")


def blind_retry_outcome(txn_dict: dict) -> dict:
    """
    What would have happened if we'd just blindly retried this transaction
    immediately, no gates, no model - using the same hidden recoverability
    function the mock gateway itself uses, so this is a fair comparison, not
    a strawman built to make Reflow look good.
    """
    rng = _seeded_rng(txn_dict["txn_id"])
    true_p = true_recovery_probability(txn_dict, rng)
    succeeded = rng.random() < true_p
    return {"succeeded": succeeded, "true_probability": round(true_p, 4)}


def reflow_vs_blind(transactions: list, retry_attempts_by_txn: dict) -> dict:
    """
    Aggregate + per-transaction comparison of Reflow's actual decisions
    against a naive "retry everything immediately" policy, computed over
    whatever's currently in the live demo dataset (as opposed to the static
    offline evaluation.metrics_output.json, which is measured once against
    the held-out 1200-row set and doesn't reflect the live demo's own data).
    """
    rows = []
    reflow_recovered = 0.0
    blind_recovered = 0.0
    blind_penalty = 0.0
    reflow_unnecessary_retries = 0
    blind_unnecessary_retries = 0

    for t in transactions:
        d = _txn_to_dict(t)
        blind = blind_retry_outcome(d)
        attempts = retry_attempts_by_txn.get(t.id, [])
        reflow_action = attempts[-1].action if attempts else None
        reflow_succeeded = t.status == "recovered"

        if reflow_succeeded:
            reflow_recovered += t.amount
        if reflow_action in (Action.STOP_NO_RETRY.value, Action.SUGGEST_ALT_METHOD.value):
            # Reflow chose not to blindly retry here; if blind retry would
            # have failed anyway, that's a correctly-avoided wasted attempt
            if not blind["succeeded"]:
                reflow_unnecessary_retries += 0  # avoided, not incurred
        elif reflow_action is None and t.status in OPEN_STATUSES:
            pass  # not yet decided

        if blind["succeeded"]:
            blind_recovered += t.amount
        else:
            blind_penalty += COST_PER_WASTED_RETRY
            blind_unnecessary_retries += 1

        rows.append(
            {
                "txn_id": t.txn_id,
                "amount": t.amount,
                "failure_reason": t.failure_reason,
                "reflow_action": reflow_action,
                "reflow_outcome": "recovered" if reflow_succeeded else t.status,
                "blind_retry_would_succeed": blind["succeeded"],
                "blind_retry_true_probability": blind["true_probability"],
            }
        )

    n = len(transactions) or 1
    return {
        "n_transactions": len(transactions),
        "reflow": {
            "revenue_recovered": round(reflow_recovered, 2),
            "recovery_rate": round(sum(1 for t in transactions if t.status == "recovered") / n, 4),
        },
        "blind_retry": {
            "revenue_recovered": round(blind_recovered, 2),
            "recovery_rate": round(sum(1 for r in rows if r["blind_retry_would_succeed"]) / n, 4),
            "penalty_cost_rupees": round(blind_penalty, 2),
            "unnecessary_retries": blind_unnecessary_retries,
        },
        "delta": {
            "revenue_recovered_diff": round(reflow_recovered - blind_recovered, 2),
            "penalty_avoided_rupees": round(blind_penalty, 2),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# 3. WhatsApp-style recovery preview (Hinglish, templated - no real API)
# ---------------------------------------------------------------------------

_WHATSAPP_TEMPLATES = {
    "insufficient_funds": (
        "Namaste! Aapka payment of ₹{amount:.0f} process nahi ho paya - lagta hai balance kam tha. "
        "Jab ready ho, is link se dobara try kar sakte hain: {retry_hint}. Koi help chahiye toh reply karein 🙂"
    ),
    "card_expired": (
        "Hi! Aapka card jo aap use kar rahe the woh expire ho chuka hai, isliye ₹{amount:.0f} ka payment fail ho gaya. "
        "Naya card add karke dobara try karein: {retry_hint}"
    ),
    "invalid_otp": (
        "Aapka payment of ₹{amount:.0f} OTP mismatch ki wajah se fail ho gaya. Koi baat nahi, dobara try karein aur is baar "
        "OTP carefully enter karein: {retry_hint}"
    ),
    "issuer_declined": (
        "Aapka bank ne is payment (₹{amount:.0f}) ko decline kar diya. Aap UPI se try kar sakte hain, usually zyada smoothly chalta hai: {retry_hint}"
    ),
}

_DEFAULT_TEMPLATE = (
    "Aapka payment of ₹{amount:.0f} complete nahi ho paya. Hum isko dekh rahe hain - jald hi update denge, ya aap dobara "
    "try kar sakte hain: {retry_hint}"
)


def whatsapp_preview(txn_dict: dict) -> dict | None:
    """
    Only makes sense for customer-side failures - a customer can't do
    anything about a bank server outage, so we don't message them for those.
    """
    reason = txn_dict["failure_reason"]
    if _classify(reason) != "customer":
        return None
    template = _WHATSAPP_TEMPLATES.get(reason, _DEFAULT_TEMPLATE)
    retry_hint = f"razorpay.me/retry/{txn_dict['txn_id']}"
    message = template.format(amount=txn_dict["amount"], retry_hint=retry_hint)
    return {"channel": "whatsapp", "language": "hinglish", "message": message}


# ---------------------------------------------------------------------------
# 4. Live policy lab - simulate different threshold/cooldown, no persistence
# ---------------------------------------------------------------------------

def simulate_policy(transactions: list, confidence_threshold: float, cooldown_multiplier: float) -> dict:
    """
    Re-runs every transaction through a parameterized version of the gate
    logic in retry_engine._decide_action, then samples the outcome from the
    same hidden ground-truth function the mock gateway uses (deterministic
    per txn_id, so dragging the slider back to the same spot reproduces the
    same numbers). Purely computed in memory - never touches the database.
    """
    n_retried = 0
    n_escalated = 0
    n_stopped_or_redirected = 0
    revenue_recovered = 0.0
    revenue_at_risk = 0.0
    penalty_cost = 0.0
    unnecessary_retries = 0

    for t in transactions:
        d = _txn_to_dict(t)
        rng = _seeded_rng(t.txn_id)

        if d["is_flagged_risk"] or d["failure_reason"] in NO_AUTO_RETRY_REASONS or d["attempt_no"] > MAX_AUTO_RETRIES:
            n_escalated += 1 if (d["is_flagged_risk"] or d["attempt_no"] > MAX_AUTO_RETRIES) else 0
            n_stopped_or_redirected += 1 if d["failure_reason"] in NO_AUTO_RETRY_REASONS else 0
            revenue_at_risk += t.amount
            continue

        if d["amount"] > LARGE_TXN_THRESHOLD and d["attempt_no"] > 1:
            n_escalated += 1
            revenue_at_risk += t.amount
            continue

        proba = predict_retry_success_proba(to_feature_dict(d))
        cooldown = COOLDOWN_MINUTES_BY_REASON.get(d["failure_reason"], DEFAULT_COOLDOWN_MINUTES) * cooldown_multiplier
        too_soon = d["minutes_since_last_failure"] < cooldown

        if proba < confidence_threshold:
            n_stopped_or_redirected += 1
            revenue_at_risk += t.amount
            continue

        if too_soon:
            revenue_at_risk += t.amount
            continue

        n_retried += 1
        outcome = blind_retry_outcome(d)  # same hidden-truth sampling, deterministic per txn
        if outcome["succeeded"]:
            revenue_recovered += t.amount
        else:
            penalty_cost += COST_PER_WASTED_RETRY
            unnecessary_retries += 1
            revenue_at_risk += t.amount

    total = len(transactions) or 1
    resolved = n_retried + n_stopped_or_redirected + n_escalated
    return {
        "confidence_threshold": confidence_threshold,
        "cooldown_multiplier": cooldown_multiplier,
        "n_total": len(transactions),
        "n_retried": n_retried,
        "n_escalated": n_escalated,
        "n_stopped_or_redirected": n_stopped_or_redirected,
        "revenue_recovered": round(revenue_recovered, 2),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "recovery_rate": round(revenue_recovered / (revenue_recovered + revenue_at_risk), 4)
        if (revenue_recovered + revenue_at_risk) else 0.0,
        "unnecessary_retries": unnecessary_retries,
        "penalty_cost_rupees": round(penalty_cost, 2),
        "escalation_rate": round(n_escalated / total, 4),
    }


# ---------------------------------------------------------------------------
# 5. Model calibration
# ---------------------------------------------------------------------------

CALIBRATION_BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def calibration_curve(retry_attempts: list) -> list[dict]:
    """Buckets predicted-success-probability against actual outcome for every
    retry attempt that ran through the model and has a resolved outcome."""
    buckets = []
    for lo, hi in CALIBRATION_BUCKETS:
        in_bucket = [
            a for a in retry_attempts
            if a.predicted_success_proba is not None
            and lo <= a.predicted_success_proba < hi
            and a.outcome in ("captured", "failed")
        ]
        n = len(in_bucket)
        actual_rate = (sum(1 for a in in_bucket if a.outcome == "captured") / n) if n else None
        avg_predicted = (sum(a.predicted_success_proba for a in in_bucket) / n) if n else None
        buckets.append(
            {
                "bucket": f"{int(lo*100)}-{int(min(hi,1.0)*100)}%",
                "n": n,
                "avg_predicted": round(avg_predicted, 3) if avg_predicted is not None else None,
                "actual_success_rate": round(actual_rate, 3) if actual_rate is not None else None,
            }
        )
    return buckets


# ---------------------------------------------------------------------------
# 6. Revenue leak radar
# ---------------------------------------------------------------------------

def revenue_leak_radar(transactions: list) -> dict:
    at_risk = [t for t in transactions if t.status in OPEN_STATUSES + ("given_up",)]

    def _breakdown(keyfn):
        agg = defaultdict(lambda: {"amount": 0.0, "count": 0})
        for t in at_risk:
            k = keyfn(t)
            agg[k]["amount"] += t.amount
            agg[k]["count"] += 1
        return sorted(
            [{"key": k, "amount": round(v["amount"], 2), "count": v["count"]} for k, v in agg.items()],
            key=lambda r: r["amount"],
            reverse=True,
        )

    return {
        "total_at_risk": round(sum(t.amount for t in at_risk), 2),
        "by_failure_reason": _breakdown(lambda t: t.failure_reason),
        "by_bank": _breakdown(lambda t: t.bank),
        "by_payment_method": _breakdown(lambda t: t.payment_method),
        "by_category": _breakdown(lambda t: _classify(t.failure_reason)),
    }


# ---------------------------------------------------------------------------
# 7. Recovery opportunity score
# ---------------------------------------------------------------------------

def opportunity_score(transactions: list) -> list[dict]:
    open_txns = [t for t in transactions if t.status in OPEN_STATUSES]
    scored = []
    for t in open_txns:
        d = _txn_to_dict(t)
        try:
            proba = predict_retry_success_proba(to_feature_dict(d))
        except Exception:
            proba = 0.0
        friction = min(0.3, 0.1 * max(0, t.attempt_no - 1))
        expected_value = t.amount * proba * (1 - friction)

        if expected_value > 2000 and proba > 0.4:
            priority = "high"
            why = f"High value (₹{t.amount:.0f}) with strong predicted success ({proba:.0%})."
        elif expected_value > 500:
            priority = "medium"
            why = f"Moderate expected recovery value (₹{expected_value:.0f})."
        else:
            priority = "low"
            why = f"Low expected value - either small amount or weak predicted success ({proba:.0%})."

        scored.append(
            {
                "txn_id": t.txn_id,
                "amount": t.amount,
                "failure_reason": t.failure_reason,
                "predicted_success_proba": round(proba, 4),
                "attempt_no": t.attempt_no,
                "expected_value": round(expected_value, 2),
                "priority": priority,
                "why": why,
            }
        )
    scored.sort(key=lambda r: r["expected_value"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# 8. Failure DNA fingerprint
# ---------------------------------------------------------------------------

def failure_dna(txn, latest_attempt=None) -> dict:
    d = _txn_to_dict(txn)
    try:
        proba = predict_retry_success_proba(to_feature_dict(d))
    except Exception:
        proba = None
    return {
        "txn_id": txn.txn_id,
        "failure_reason": txn.failure_reason,
        "payment_method": txn.payment_method,
        "bank": txn.bank,
        "attempt_no": txn.attempt_no,
        "classification": _classify(txn.failure_reason),
        "diagnosis": diagnose(d),
        "recovery_likelihood": round(proba, 4) if proba is not None else None,
        "current_action": latest_attempt.action if latest_attempt else None,
        "current_reason_code": latest_attempt.reason_code if latest_attempt else None,
    }


# ---------------------------------------------------------------------------
# 9. Chaos / incident simulator - synthetic, never persisted
# ---------------------------------------------------------------------------

@dataclass
class ChaosScenario:
    key: str
    label: str
    bank: str | None
    payment_method: str | None
    failure_reason: str
    count: int


CHAOS_SCENARIOS = {
    "issuer_degradation": ChaosScenario("issuer_degradation", "Issuer degradation (HDFC declines spike)", "hdfc", None, "issuer_declined", 8),
    "upi_outage": ChaosScenario("upi_outage", "UPI rail outage", None, "upi", "gateway_error", 10),
    "network_latency": ChaosScenario("network_latency", "Elevated network latency", None, None, "network_timeout", 7),
    "decline_spike": ChaosScenario("decline_spike", "Sudden decline spike (SBI)", "sbi", None, "issuer_declined", 9),
}


def inject_chaos_scenario(scenario_key: str, base_amount: float = 1500.0):
    """
    Builds a batch of synthetic in-memory transaction-like objects for a
    named failure scenario. These are plain objects (not ORM rows) shaped
    like Transaction for detect_incidents() to consume - nothing here is
    written to the database.
    """
    scenario = CHAOS_SCENARIOS.get(scenario_key)
    if not scenario:
        return None, []

    class _FakeTxn:
        pass

    injected = []
    banks = ["hdfc", "sbi", "icici", "axis", "yesbank"]
    methods = ["card", "upi", "netbanking", "wallet"]
    rng = random.Random(f"chaos::{scenario_key}")
    for i in range(scenario.count):
        f = _FakeTxn()
        f.txn_id = f"chaos_{scenario_key}_{i}"
        f.amount = round(base_amount * rng.uniform(0.5, 2.5), 2)
        f.payment_method = scenario.payment_method or rng.choice(methods)
        f.bank = scenario.bank or rng.choice(banks)
        f.failure_reason = scenario.failure_reason
        f.attempt_no = 1
        f.hour_of_day = rng.randint(0, 23)
        f.is_weekend = 0
        f.customer_past_success_rate = round(rng.uniform(0.3, 0.9), 2)
        f.minutes_since_last_failure = rng.randint(0, 20)
        f.is_flagged_risk = 0
        f.status = "failed"
        injected.append(f)
    return scenario, injected
