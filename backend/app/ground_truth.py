"""
The hidden "does this retry actually clear" function.

This is the one place that defines how a failed payment's real-world
recoverability works - bank outages clearing after enough time has passed,
large transactions facing more issuer scrutiny, UPI clearing slightly more
often than card, and so on. Both `data/generate_synthetic_data.py` (which
labels the training/eval data) and `app/razorpay_mock.py` (which decides
whether a *live* retry succeeds in the demo) call into this same function,
so there's exactly one definition of "truth" instead of two copies that can
drift apart.

Nothing in `app/ml` or `app/retry_engine` imports this module, and it never
sees the model's prediction. That's the point: the model's job is to guess
this function from the features it's given, and the mock gateway's job is
to apply the real thing, independently, regardless of what the model
guessed. If the mock gateway used the model's own score to decide whether
a retry succeeds, every "high confidence" prediction would inflate its own
success rate and the evaluation would be worthless - the model would look
good for the sole reason that it was allowed to grade its own homework.
"""
import random

# base recoverability by failure reason, before any of the situational
# adjustments below - mirrors how these failures actually behave in the
# field (a bank outage clears on its own most of the time, an expired card
# never does)
FAILURE_REASON_BASE_RECOVERABILITY = {
    "insufficient_funds": 0.35,
    "bank_server_down": 0.78,
    "network_timeout": 0.72,
    "card_expired": 0.03,
    "invalid_otp": 0.55,
    "issuer_declined": 0.20,
    "risk_block": 0.02,
    "gateway_error": 0.68,
}


def true_recovery_probability(txn: dict, rng: random.Random) -> float:
    """
    The real (hidden) chance this specific failed payment clears if retried
    right now, given its actual circumstances. `txn` needs: failure_reason,
    payment_method, hour_of_day, attempt_no, customer_past_success_rate,
    minutes_since_last_failure, amount. `rng` is caller-owned so callers
    control their own reproducibility/seeding.
    """
    reason = txn["failure_reason"]
    method = txn["payment_method"]

    p = FAILURE_REASON_BASE_RECOVERABILITY.get(reason, 0.4)

    if method == "upi":
        p += 0.05
    if method == "netbanking" and reason == "bank_server_down":
        p += 0.10
    if txn.get("hour_of_day") in (1, 2, 3, 4):  # bank maintenance windows, common in India
        p -= 0.15
    if txn.get("attempt_no", 1) >= 3:
        p -= 0.20  # diminishing returns on repeated retries

    p += (txn.get("customer_past_success_rate", 0.5) - 0.5) * 0.3

    minutes = txn.get("minutes_since_last_failure", 0)
    if minutes < 5 and reason in ("bank_server_down", "network_timeout"):
        p -= 0.12  # retrying instantly during an outage rarely helps
    if minutes >= 30 and reason in ("bank_server_down", "network_timeout"):
        p += 0.10

    if txn.get("amount", 0) > 50000:
        p -= 0.08  # large txns face more issuer scrutiny

    p += rng.gauss(0, 0.06)  # noise - nothing in this domain is perfectly deterministic
    return min(0.97, max(0.01, p))
