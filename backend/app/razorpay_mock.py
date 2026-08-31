"""
Stand-in for the real Razorpay Orders/Payments API in test mode. We don't
have a live Razorpay test account wired into this environment, so this
simulates the same request/response shape their API uses (payment_id,
order_id, status transitions).

Outcomes are sampled from `app.ground_truth.true_recovery_probability` -
the same hidden function used to label the synthetic training/eval data -
not from whatever the retry engine's model predicted for this transaction.
That's deliberate: if the mock gateway used the model's own score to decide
whether a retry clears, a "confident" prediction would inflate its own
success rate and every metric downstream of it (recovery rate, revenue
recovered, the dashboard) would be measuring the model agreeing with
itself rather than measuring anything real. The gateway doesn't know or
care what the model guessed; it just applies the real (hidden) odds for
this transaction's actual circumstances.

Swapping this for the real SDK is a one-file change: `RazorpayClient` below
has the same method signature `retry_payment(order_id, amount, method, ...)`
that `razorpay.Client().payment.capture(...)` style calls would need, plus
a `context` dict carrying the recoverability signals this mock uses that a
real gateway obviously wouldn't need from the caller.
"""
import random
import time
import uuid

from app.ground_truth import true_recovery_probability


class RazorpayMockError(Exception):
    pass


class RazorpayClient:
    def __init__(self, key_id: str = "rzp_test_mock", key_secret: str = "mock_secret", seed: int | None = None):
        self.key_id = key_id
        self.key_secret = key_secret
        self._rng = random.Random(seed)

    def retry_payment(self, order_id: str, amount: float, method: str, context: dict) -> dict:
        """
        Simulates re-attempting a payment. `context` carries the
        transaction's real-world recoverability signals (failure_reason,
        attempt_no, hour_of_day, customer_past_success_rate,
        minutes_since_last_failure, amount) - the caller's job is just to
        pass through what it knows about the transaction, not to tell this
        client whether it thinks the retry will work.
        """
        time.sleep(0.05)  # pretend there's a network hop

        if self._rng.random() < 0.02:
            raise RazorpayMockError("gateway_timeout: upstream did not respond in time")

        true_p = true_recovery_probability(context, self._rng)
        succeeded = self._rng.random() < true_p
        payment_id = f"pay_mock_{uuid.uuid4().hex[:14]}"

        return {
            "id": payment_id,
            "order_id": order_id,
            "amount": int(amount * 100),  # paise, matching real API
            "currency": "INR",
            "method": method,
            "status": "captured" if succeeded else "failed",
            "error_code": None if succeeded else self._rng.choice(
                ["BAD_REQUEST_ERROR", "GATEWAY_ERROR", "SERVER_ERROR"]
            ),
        }
