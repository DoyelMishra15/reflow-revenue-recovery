import os
import joblib
import pandas as pd

from app.ml.features import FEATURE_COLS

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
_model = None


def _get_model():
    global _model
    if _model is None:
        if not os.path.exists(_MODEL_PATH):
            raise RuntimeError(
                "model.pkl not found - run `python -m app.ml.train_model` from backend/ first"
            )
        _model = joblib.load(_MODEL_PATH)
    return _model


def predict_retry_success_proba(features: dict) -> float:
    """features must have all keys in FEATURE_COLS."""
    row = pd.DataFrame([{c: features[c] for c in FEATURE_COLS}])
    model = _get_model()
    return float(model.predict_proba(row)[:, 1][0])
