import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retry_engine import decide, Action, MAX_AUTO_RETRIES


def base_txn(**overrides):
    txn = {
        "amount": 500.0,
        "payment_method": "upi",
        "bank": "hdfc",
        "failure_reason": "bank_server_down",
        "attempt_no": 1,
        "hour_of_day": 14,
        "is_weekend": 0,
        "customer_past_success_rate": 0.8,
        "minutes_since_last_failure": 60,
        "is_flagged_risk": 0,
    }
    txn.update(overrides)
    return txn


def test_risk_flag_always_escalates_never_retries():
    txn = base_txn(is_flagged_risk=1, failure_reason="issuer_declined")
    d = decide(txn)
    assert d.action == Action.ESCALATE_HUMAN
    assert "risk_flag" in d.gates_triggered


def test_card_expired_never_auto_retried():
    txn = base_txn(failure_reason="card_expired")
    d = decide(txn)
    assert d.action == Action.STOP_NO_RETRY
    assert d.predicted_success_proba is None  # gate stops it before the model even runs


def test_max_retries_gate_stops_further_attempts():
    txn = base_txn(attempt_no=MAX_AUTO_RETRIES + 1)
    d = decide(txn)
    assert d.action == Action.ESCALATE_HUMAN
    assert "max_retries_exceeded" in d.gates_triggered


def test_large_txn_repeat_attempt_escalates_instead_of_auto_retrying():
    txn = base_txn(amount=200_000, attempt_no=2, failure_reason="network_timeout")
    d = decide(txn)
    assert d.action == Action.ESCALATE_HUMAN
    assert "large_txn_repeat_attempt" in d.gates_triggered


def test_large_txn_first_attempt_is_not_gated_by_amount_alone():
    txn = base_txn(amount=200_000, attempt_no=1, failure_reason="network_timeout", minutes_since_last_failure=60)
    d = decide(txn)
    # should reach the model, not get blocked purely for being large on attempt 1
    assert "large_txn_repeat_attempt" not in d.gates_triggered


def test_cooldown_delays_retry_even_with_good_prediction():
    txn = base_txn(failure_reason="bank_server_down", minutes_since_last_failure=1, customer_past_success_rate=0.95)
    d = decide(txn)
    assert d.action in (Action.DELAY_RETRY, Action.STOP_NO_RETRY)
    if d.action == Action.DELAY_RETRY:
        assert d.cooldown_minutes is not None and d.cooldown_minutes > 0


def test_decision_always_returns_explanation():
    for reason in ["insufficient_funds", "bank_server_down", "card_expired", "risk_block"]:
        d = decide(base_txn(failure_reason=reason, is_flagged_risk=1 if reason == "risk_block" else 0))
        assert d.explanation
        assert isinstance(d.explanation, str)


def test_every_action_is_bounded_to_known_enum():
    txn = base_txn()
    d = decide(txn)
    assert d.action in list(Action)


def test_diagnosis_is_attached_regardless_of_which_gate_fired():
    # gate path (risk flag) and model path (clean bank_server_down) should
    # both come back with a diagnosis - it's not just tacked onto the happy path
    gated = decide(base_txn(is_flagged_risk=1, failure_reason="risk_block"))
    modeled = decide(base_txn(failure_reason="bank_server_down"))
    assert gated.diagnosis == "fraud/risk signal"
    assert modeled.diagnosis == "bank-side, infra outage"
