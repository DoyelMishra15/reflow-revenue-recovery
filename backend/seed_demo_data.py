"""
Loads a slice of the synthetic dataset as 'live' failed transactions so the
dashboard has something to show right after a fresh clone. Doesn't touch the
retry_succeeded column - that's the held-out ground truth the mock Razorpay
client uses internally, not something the app gets to see upfront.

Usage: python seed_demo_data.py [--n 40] [--api http://localhost:8000]
"""
import argparse
import csv
import os
import sys

import httpx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "transactions_sample.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    with open(DATA_PATH) as f:
        rows = list(csv.DictReader(f))[: args.n]

    ok, skipped = 0, 0
    for row in rows:
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
        r = httpx.post(f"{args.api}/transactions", json=payload, timeout=10)
        if r.status_code == 200:
            ok += 1
        else:
            skipped += 1

    print(f"seeded {ok} transactions, skipped {skipped} (likely already existed)")


if __name__ == "__main__":
    sys.exit(main())
