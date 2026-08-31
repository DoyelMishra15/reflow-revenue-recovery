"""
Compares the retry engine's decisions against two naive baselines on the
held-out portion of the synthetic dataset:

  - retry_all: always retry immediately, no gates, no model
  - retry_all_gated: same gates as our engine (risk/expired/max-attempts)
    but no ML score - retries anything that clears the gates

This isolates what the model actually buys us on top of the rule gates
alone, since the gates already do a lot of the work. That's the honest
comparison - not retry_all vs us, since retry_all is a strawman.

Because we have ground truth `retry_succeeded` for the synthetic data
(which the live system never sees), we can score each policy exactly:
for every txn, would this policy have retried it, and if so did that
retry actually succeed (per ground truth)?

Also reports the classifier's own confusion matrix / precision / recall / F1
at the operating threshold actually used in production (MIN_PROBA_TO_RETRY),
not just the 0.5 default reported by train_model.py - those are two different
questions ("is the model well-calibrated" vs "is the threshold we ship any
good") and conflating them is a common way buildathon evals paper over a bad
threshold choice.

Run (from backend/): python -m evaluation.evaluate
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.retry_engine import (  # noqa: E402
    decide,
    Action,
    NO_AUTO_RETRY_REASONS,
    MAX_AUTO_RETRIES,
    MIN_PROBA_TO_RETRY,
)
from app.ml.classifier import predict_retry_success_proba  # noqa: E402
from app.ml.features import to_feature_dict  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "transactions_sample.csv")
OUT_PATH = os.path.join(HERE, "metrics_output.json")

RETRY_ACTIONS = {Action.RETRY_NOW.value, Action.DELAY_RETRY.value}
ESCALATE_ACTIONS = {Action.ESCALATE_HUMAN.value}
AVOIDED_ACTIONS = {Action.STOP_NO_RETRY.value, Action.SUGGEST_ALT_METHOD.value}

# what it costs, in rupees, every time we retry something that fails anyway -
# not the txn amount (nothing is lost, the txn was already failed), but the
# operational cost of a wasted gateway attempt: issuer-side fraud-scoring
# friction, gateway processing cost, customer annoyance. Made explicit and
# tunable rather than left as an unstated assumption baked into "wasted_retries".
COST_PER_WASTED_RETRY = 8.0


def policy_retry_all(row) -> str:
    return "retry"


def policy_retry_all_gated(row) -> str:
    if row["is_flagged_risk"]:
        return "escalate"
    if row["failure_reason"] in NO_AUTO_RETRY_REASONS:
        return "stop"
    if row["attempt_no"] > MAX_AUTO_RETRIES:
        return "escalate"
    return "retry"


def policy_reflow(row) -> str:
    d = decide(row.to_dict())
    if d.action.value in RETRY_ACTIONS:
        return "retry"
    if d.action.value in ESCALATE_ACTIONS:
        return "escalate"
    return "stop"  # stop_no_retry / suggest_alt_method - no gateway call made


def score_policy(df: pd.DataFrame, policy_fn) -> dict:
    decisions = df.apply(policy_fn, axis=1)
    would_retry = decisions == "retry"
    n_retried = int(would_retry.sum())
    n_escalated = int((decisions == "escalate").sum())
    n_stopped = int((decisions == "stop").sum())
    n_total = len(df)

    retried = df[would_retry]
    successes = int(retried["retry_succeeded"].sum())
    wasted = n_retried - successes  # retries that were attempted but failed anyway

    # revenue at risk = every failed txn's amount, before any policy acts on it
    revenue_at_risk = float(df["amount"].sum())
    revenue_recovered = float(retried.loc[retried["retry_succeeded"] == 1, "amount"].sum())

    # false positives here = txns the policy retried that ground truth says
    # would NOT have succeeded - the "unnecessary action" the eval is scored on
    false_positive_retries = wasted
    false_positive_cost = round(false_positive_retries * COST_PER_WASTED_RETRY, 2)

    success_rate_when_retried = successes / n_retried if n_retried else 0.0

    return {
        "n_total": n_total,
        "n_retried": n_retried,
        "n_escalated": n_escalated,
        "n_stopped_or_redirected": n_stopped,
        "retry_rate": round(n_retried / n_total, 4),
        "escalation_rate": round(n_escalated / n_total, 4),
        "successful_retries": successes,
        "unnecessary_retries": wasted,
        "unnecessary_retry_rate": round(wasted / n_retried, 4) if n_retried else 0.0,
        "success_rate_when_retried": round(success_rate_when_retried, 4),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "revenue_recovered": round(revenue_recovered, 2),
        "recovery_rate_of_at_risk": round(revenue_recovered / revenue_at_risk, 4) if revenue_at_risk else 0.0,
        "false_positive_cost_rupees": false_positive_cost,
    }


def classifier_confusion_matrix(test_df: pd.DataFrame, threshold: float) -> dict:
    """
    Confusion matrix for the underlying classifier's own predict-retry-success
    call, independent of the gates around it, at the threshold actually used
    in production (MIN_PROBA_TO_RETRY). Positive class = retry succeeds.
    """
    probas = test_df.apply(lambda r: predict_retry_success_proba(to_feature_dict(r.to_dict())), axis=1)
    preds = (probas >= threshold).astype(int)
    actual = test_df["retry_succeeded"].astype(int)

    tp = int(((preds == 1) & (actual == 1)).sum())
    fp = int(((preds == 1) & (actual == 0)).sum())
    tn = int(((preds == 0) & (actual == 0)).sum())
    fn = int(((preds == 0) & (actual == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": threshold,
        "confusion_matrix": {"true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn},
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(test_df), 4) if len(test_df) else 0.0,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    # evaluate on the same held-out 20% split used to test the classifier, so
    # this is never scored on rows the model trained on
    from sklearn.model_selection import train_test_split

    _, test_df = train_test_split(df, test_size=0.2, random_state=7, stratify=df["retry_succeeded"])

    results = {
        "retry_all": score_policy(test_df, policy_retry_all),
        "retry_all_gated": score_policy(test_df, policy_retry_all_gated),
        "reflow_ml_policy": score_policy(test_df, policy_reflow),
        "classifier_at_production_threshold": classifier_confusion_matrix(test_df, MIN_PROBA_TO_RETRY),
    }

    gated = results["retry_all_gated"]
    reflow = results["reflow_ml_policy"]
    results["uplift_vs_gated_baseline"] = {
        "unnecessary_retries_reduced_by": gated["unnecessary_retries"] - reflow["unnecessary_retries"],
        "unnecessary_retries_reduced_pct": round(
            100 * (gated["unnecessary_retries"] - reflow["unnecessary_retries"]) / max(gated["unnecessary_retries"], 1), 1
        ),
        "false_positive_cost_saved_rupees": round(
            gated["false_positive_cost_rupees"] - reflow["false_positive_cost_rupees"], 2
        ),
        "success_rate_when_retried_delta": round(
            reflow["success_rate_when_retried"] - gated["success_rate_when_retried"], 4
        ),
        "revenue_recovered_delta": round(reflow["revenue_recovered"] - gated["revenue_recovered"], 2),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nwritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
