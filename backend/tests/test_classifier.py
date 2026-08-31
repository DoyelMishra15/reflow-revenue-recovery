import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ml.classifier import predict_retry_success_proba


def test_proba_is_in_valid_range():
    features = {
        "amount": 500.0,
        "attempt_no": 1,
        "hour_of_day": 14,
        "is_weekend": 0,
        "customer_past_success_rate": 0.8,
        "minutes_since_last_failure": 60,
        "is_flagged_risk": 0,
        "payment_method": "upi",
        "bank": "hdfc",
        "failure_reason": "bank_server_down",
    }
    p = predict_retry_success_proba(features)
    assert 0.0 <= p <= 1.0


def test_card_expired_scores_lower_than_bank_server_down():
    base = {
        "amount": 500.0,
        "attempt_no": 1,
        "hour_of_day": 14,
        "is_weekend": 0,
        "customer_past_success_rate": 0.8,
        "minutes_since_last_failure": 60,
        "is_flagged_risk": 0,
        "payment_method": "card",
        "bank": "hdfc",
    }
    p_expired = predict_retry_success_proba({**base, "failure_reason": "card_expired"})
    p_server_down = predict_retry_success_proba({**base, "failure_reason": "bank_server_down"})
    assert p_expired < p_server_down


def test_higher_past_success_rate_increases_proba():
    base = {
        "amount": 500.0,
        "attempt_no": 1,
        "hour_of_day": 14,
        "is_weekend": 0,
        "minutes_since_last_failure": 60,
        "is_flagged_risk": 0,
        "payment_method": "upi",
        "bank": "hdfc",
        "failure_reason": "insufficient_funds",
    }
    p_low = predict_retry_success_proba({**base, "customer_past_success_rate": 0.1})
    p_high = predict_retry_success_proba({**base, "customer_past_success_rate": 0.95})
    assert p_high > p_low
