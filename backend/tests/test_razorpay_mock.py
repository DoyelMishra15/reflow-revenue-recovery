import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from unittest.mock import patch

import pytest

from app.razorpay_mock import RazorpayClient, RazorpayMockError
from app.ground_truth import true_recovery_probability
import random


@pytest.fixture(autouse=True)
def _no_sleep():
    # these tests make hundreds of calls to exercise the outcome
    # distribution - skip the mock's simulated network delay so the suite
    # stays fast; doesn't touch the sampling logic itself
    with patch("app.razorpay_mock.time.sleep", lambda *_: None):
        yield


def _context(**overrides):
    ctx = {
        "amount": 500.0,
        "payment_method": "upi",
        "failure_reason": "bank_server_down",
        "attempt_no": 1,
        "hour_of_day": 14,
        "customer_past_success_rate": 0.8,
        "minutes_since_last_failure": 60,
    }
    ctx.update(overrides)
    return ctx


def test_retry_payment_has_no_success_bias_param():
    """
    The whole point of the fix: the mock client can't be handed a
    model-derived probability to sample from. If this signature ever grows
    a `success_bias` (or similarly named) parameter again, that's the old
    circular bug creeping back in.
    """
    import inspect

    sig = inspect.signature(RazorpayClient.retry_payment)
    assert "success_bias" not in sig.parameters
    assert "context" in sig.parameters


def test_outcomes_track_ground_truth_not_a_fixed_bias():
    """
    A transaction with a genuinely poor hidden recoverability (near-certain
    to fail, per app.ground_truth) should mostly fail against the mock
    gateway, and a genuinely strong one should mostly succeed - regardless
    of what any caller might believe about it. This is what makes the mock
    gateway usable as ground truth for evaluation instead of an echo
    chamber.
    """
    client = RazorpayClient(seed=123)

    weak_ctx = _context(failure_reason="card_expired", customer_past_success_rate=0.05, amount=200000)
    strong_ctx = _context(
        failure_reason="bank_server_down",
        payment_method="netbanking",
        customer_past_success_rate=0.95,
        minutes_since_last_failure=60,
        amount=300,
    )

    weak_successes = 0
    strong_successes = 0
    n = 200
    for i in range(n):
        try:
            r = client.retry_payment(order_id=f"o{i}", amount=weak_ctx["amount"], method=weak_ctx["payment_method"], context=weak_ctx)
        except RazorpayMockError:
            continue
        if r["status"] == "captured":
            weak_successes += 1
    for i in range(n):
        try:
            r = client.retry_payment(order_id=f"o{i}", amount=strong_ctx["amount"], method=strong_ctx["payment_method"], context=strong_ctx)
        except RazorpayMockError:
            continue
        if r["status"] == "captured":
            strong_successes += 1

    # weak context should rarely clear, strong context should usually clear -
    # loose bounds since there's a small chance of a simulated gateway_timeout
    # exception being raised instead of a captured/failed result
    assert weak_successes / n < 0.15
    assert strong_successes / n > 0.65


def test_same_context_gives_different_outcomes_across_calls():
    """
    Outcomes are sampled, not deterministic per-context - a context with a
    ~50% true recovery probability shouldn't resolve to the same status
    every single time.
    """
    client = RazorpayClient(seed=7)
    ctx = _context(failure_reason="invalid_otp", customer_past_success_rate=0.5)
    statuses = set()
    for i in range(30):
        try:
            r = client.retry_payment(order_id=f"o{i}", amount=ctx["amount"], method=ctx["payment_method"], context=ctx)
        except RazorpayMockError:
            continue
        statuses.add(r["status"])
    assert statuses == {"captured", "failed"}


def test_mock_outcome_probability_matches_ground_truth_function():
    """
    Sanity check that the mock gateway's long-run success rate for a fixed
    context lines up with app.ground_truth.true_recovery_probability for
    that same context (within sampling noise) - confirming it's actually
    that function driving outcomes, not some other hardcoded rate.
    """
    ctx = _context(failure_reason="gateway_error", customer_past_success_rate=0.7, minutes_since_last_failure=45)
    rng = random.Random(99)
    expected = sum(true_recovery_probability(ctx, rng) for _ in range(500)) / 500

    client = RazorpayClient(seed=99)
    n = 500
    successes = 0
    attempted = 0
    for i in range(n):
        try:
            r = client.retry_payment(order_id=f"o{i}", amount=ctx["amount"], method=ctx["payment_method"], context=ctx)
        except RazorpayMockError:
            continue
        attempted += 1
        if r["status"] == "captured":
            successes += 1
    observed = successes / attempted

    assert abs(observed - expected) < 0.1
