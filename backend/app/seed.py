"""
Demo-data auto-seeding.

Ingests a fixed set of transactions directly through the DB session (no HTTP
round-trip to itself) so a freshly deployed / freshly restarted instance
always has something useful on the dashboard, without depending on anyone
running `seed_demo_data.py` by hand against their own Codespace.

Two parts:

1. GUARANTEED_DEMO_TXNS - a handful of hand-picked transactions, one per
   recovery action, so a judge can reliably see all four paths
   (retry_now / stop_no_retry / escalate_human / suggest_alt_method)
   without hunting through a random sample for one that happens to trigger
   each. These are demo fixtures, chosen to land on a specific action given
   the current retry_engine gates/model - not fabricated decisions; running
   /decide on them still calls the real engine.
2. A broader slice of the synthetic dataset (data/transactions_sample.csv)
   for volume/realism, same rows `seed_demo_data.py` would load manually.

Only runs when the transactions table is empty, so it never clobbers real
activity from someone poking at the demo.
"""
import csv
import os
from datetime import datetime, timedelta, timezone

from app.db import SessionLocal
from app.models import Transaction

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "transactions_sample.csv")

GUARANTEED_DEMO_TXNS = [
    # -> retry_now: transient network blip, cooldown already cleared, decent
    # customer history - model should score this high.
    {
        "txn_id": "demo_retry_now",
        "amount": 480.0,
        "payment_method": "upi",
        "bank": "hdfc",
        "failure_reason": "network_timeout",
        "attempt_no": 1,
        "hour_of_day": 11,
        "is_weekend": 0,
        "customer_past_success_rate": 0.9,
        "minutes_since_last_failure": 40,
        "is_flagged_risk": 0,
    },
    # -> stop_no_retry: card_expired is a hard gate, model never even runs.
    {
        "txn_id": "demo_stop_no_retry",
        "amount": 1200.0,
        "payment_method": "card",
        "bank": "icici",
        "failure_reason": "card_expired",
        "attempt_no": 1,
        "hour_of_day": 14,
        "is_weekend": 0,
        "customer_past_success_rate": 0.5,
        "minutes_since_last_failure": 10,
        "is_flagged_risk": 0,
    },
    # -> escalate_human: risk flag is a hard gate, routed straight to
    # fraud review regardless of anything else about the transaction.
    {
        "txn_id": "demo_escalate_human",
        "amount": 3200.0,
        "payment_method": "card",
        "bank": "axis",
        "failure_reason": "issuer_declined",
        "attempt_no": 1,
        "hour_of_day": 16,
        "is_weekend": 0,
        "customer_past_success_rate": 0.4,
        "minutes_since_last_failure": 12,
        "is_flagged_risk": 1,
    },
    # -> suggest_alt_method: issuer decline, low predicted success on card,
    # non-UPI - engine nudges toward a different rail instead of retrying blind.
    {
        "txn_id": "demo_suggest_alt_method",
        "amount": 950.0,
        "payment_method": "card",
        "bank": "sbi",
        "failure_reason": "issuer_declined",
        "attempt_no": 1,
        "hour_of_day": 9,
        "is_weekend": 0,
        "customer_past_success_rate": 0.15,
        "minutes_since_last_failure": 10,
        "is_flagged_risk": 0,
    },
]

_TXN_FIELDS = [
    "txn_id",
    "amount",
    "payment_method",
    "bank",
    "failure_reason",
    "attempt_no",
    "hour_of_day",
    "is_weekend",
    "customer_past_success_rate",
    "minutes_since_last_failure",
    "is_flagged_risk",
]


def _insert(db, payload: dict) -> None:
    txn = Transaction(**{k: payload[k] for k in _TXN_FIELDS})
    txn.last_failure_at = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
        minutes=payload["minutes_since_last_failure"]
    )
    db.add(txn)


def seed_if_empty(sample_size: int = 40) -> int:
    """
    If the transactions table is empty, loads the guaranteed demo set plus a
    slice of the synthetic dataset. Returns the number of rows inserted (0
    if the table already had data, i.e. this was a no-op).
    """
    db = SessionLocal()
    try:
        if db.query(Transaction).count() > 0:
            return 0

        inserted = 0
        seen_ids = set()
        for payload in GUARANTEED_DEMO_TXNS:
            _insert(db, payload)
            seen_ids.add(payload["txn_id"])
            inserted += 1

        if os.path.exists(DATA_PATH):
            with open(DATA_PATH) as f:
                rows = list(csv.DictReader(f))
            for row in rows[:sample_size]:
                if row["txn_id"] in seen_ids:
                    continue
                payload = {
                    "txn_id": row["txn_id"],
                    "amount": float(row["amount"]),
                    "payment_method": row["payment_method"],
                    "bank": row["bank"],
                    "failure_reason": row["failure_reason"],
                    "attempt_no": int(row["attempt_no"]),
                    "hour_of_day": int(row["hour_of_day"]),
                    "is_weekend": int(row["is_weekend"]),
                    "customer_past_success_rate": float(row["customer_past_success_rate"]),
                    "minutes_since_last_failure": int(row["minutes_since_last_failure"]),
                    "is_flagged_risk": int(row["is_flagged_risk"]),
                }
                _insert(db, payload)
                seen_ids.add(payload["txn_id"])
                inserted += 1

        db.commit()
        return inserted
    finally:
        db.close()
