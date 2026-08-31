"""
Trains the retry-success classifier on the synthetic transaction data.

Usage (from backend/):
    python -m app.ml.train_model
"""
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.ml.features import CATEGORICAL_COLS, NUMERIC_COLS, FEATURE_COLS, TARGET_COL  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "..", "data", "transactions_sample.csv")
MODEL_PATH = os.path.join(HERE, "model.pkl")
METRICS_PATH = os.path.join(HERE, "train_metrics.json")


def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        remainder="passthrough",
    )
    clf = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.08, random_state=7
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "brier_score": round(brier_score_loss(y_test, proba), 4),
        "precision_at_0.5": round(precision_score(y_test, preds), 4),
        "recall_at_0.5": round(recall_score(y_test, preds), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "base_success_rate": round(float(y.mean()), 4),
    }

    joblib.dump(pipe, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
