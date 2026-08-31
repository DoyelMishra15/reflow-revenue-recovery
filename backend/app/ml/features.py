"""Shared feature prep so training and inference never drift apart."""

CATEGORICAL_COLS = ["payment_method", "bank", "failure_reason"]
NUMERIC_COLS = [
    "amount",
    "attempt_no",
    "hour_of_day",
    "is_weekend",
    "customer_past_success_rate",
    "minutes_since_last_failure",
    "is_flagged_risk",
]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
TARGET_COL = "retry_succeeded"


def to_feature_dict(txn: dict) -> dict:
    """Pulls just the model's feature columns out of a transaction dict/row."""
    return {c: txn[c] for c in FEATURE_COLS}
