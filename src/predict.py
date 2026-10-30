"""Load the selected model and produce validated single-row predictions."""

from functools import lru_cache
from typing import Any

import pandas as pd
from joblib import load

from src.config import FEATURE_COLUMNS, MODEL_PATH, MODEL_VERSION


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run `python scripts/run_pipeline.py` first."
        )
    return load(MODEL_PATH)


def predict_patient(features: dict[str, float]) -> dict[str, Any]:
    missing = [name for name in FEATURE_COLUMNS if name not in features]
    extra = sorted(set(features) - set(FEATURE_COLUMNS))
    if missing or extra:
        raise ValueError(f"Invalid features. Missing: {missing}; unexpected: {extra}")
    try:
        row = {name: float(features[name]) for name in FEATURE_COLUMNS}
    except (TypeError, ValueError) as exc:
        raise ValueError("All feature values must be numeric.") from exc
    model = load_model()
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0, 1])
    return {
        "prediction": prediction,
        "prediction_label": "Higher likelihood of heart disease" if prediction else "Lower likelihood of heart disease",
        "probability": probability,
        "model_version": MODEL_VERSION,
    }
