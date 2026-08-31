"""
Generates synthetic failed-payment transactions for training/evaluating the
retry-recovery model. There's no public Razorpay failure-rate dataset (for
obvious reasons), so this samples noisy features and labels each row with
`app.ground_truth.true_recovery_probability` - the same hidden "ground
truth" function the mock Razorpay client uses at runtime to decide whether
a live retry actually clears. The model never sees that function directly,
only the sampled features and the resulting outcome, same as it would with
real production data.

Run: python generate_synthetic_data.py [--n 6000] [--seed 42]
"""
import argparse
import csv
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app.ground_truth import true_recovery_probability  # noqa: E402

# (reason, sampling weight) - how often each failure reason shows up in the
# data. Base recoverability for each reason lives in app.ground_truth, the
# same place the mock gateway reads it from, so the two never drift apart.
FAILURE_REASON_WEIGHTS = [
    ("insufficient_funds", 0.22),
    ("bank_server_down", 0.15),
    ("network_timeout", 0.13),
    ("card_expired", 0.08),
    ("invalid_otp", 0.12),
    ("issuer_declined", 0.14),
    ("risk_block", 0.06),
    ("gateway_error", 0.10),
]

PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet"]
BANKS = ["hdfc", "icici", "sbi", "axis", "kotak", "yesbank", "idfc", "other"]


def sample_reason(rng):
    reasons, weights = zip(*FAILURE_REASON_WEIGHTS)
    return rng.choices(reasons, weights=weights, k=1)[0]


def make_row(rng, txn_id):
    reason = sample_reason(rng)
    method = rng.choice(PAYMENT_METHODS)
    bank = rng.choice(BANKS)
    amount = round(rng.lognormvariate(6.5, 1.1), 2)  # skewed, mostly small txns
    amount = min(amount, 250000)

    hour = rng.randint(0, 23)
    is_weekend = 1 if rng.random() < 2 / 7 else 0
    attempt_no = rng.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
    customer_past_success_rate = round(min(1.0, max(0.0, rng.gauss(0.72, 0.2))), 3)
    minutes_since_failure = rng.choice([1, 5, 15, 30, 60, 180, 720])

    # --- hidden ground truth (not exposed as a feature) ---
    # same function the mock gateway uses at runtime - see app/ground_truth.py
    p = true_recovery_probability(
        {
            "failure_reason": reason,
            "payment_method": method,
            "hour_of_day": hour,
            "attempt_no": attempt_no,
            "customer_past_success_rate": customer_past_success_rate,
            "minutes_since_last_failure": minutes_since_failure,
            "amount": amount,
        },
        rng,
    )
    retry_succeeded = 1 if rng.random() < p else 0

    is_flagged_risk = 1 if reason == "risk_block" or (amount > 150000 and rng.random() < 0.3) else 0

    ts = datetime(2026, 6, 1) + timedelta(minutes=rng.randint(0, 60 * 24 * 60))

    return {
        "txn_id": f"txn_{txn_id:06d}",
        "timestamp": ts.isoformat(),
        "amount": amount,
        "payment_method": method,
        "bank": bank,
        "failure_reason": reason,
        "attempt_no": attempt_no,
        "hour_of_day": hour,
        "is_weekend": is_weekend,
        "customer_past_success_rate": customer_past_success_rate,
        "minutes_since_last_failure": minutes_since_failure,
        "is_flagged_risk": is_flagged_risk,
        "retry_succeeded": retry_succeeded,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="transactions_sample.csv")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = [make_row(rng, i) for i in range(1, args.n + 1)]

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
