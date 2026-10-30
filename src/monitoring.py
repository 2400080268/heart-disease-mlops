"""Basic prediction log and summary statistics for classroom monitoring."""

from datetime import datetime, timezone
import json

import pandas as pd

from src.config import FEATURE_COLUMNS, METRICS_DIR, PREDICTION_LOG_PATH


def log_prediction(features: dict, result: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **features,
           "prediction": result["prediction"], "probability": result["probability"],
           "model_version": result["model_version"]}
    pd.DataFrame([row]).to_csv(
        PREDICTION_LOG_PATH, mode="a", header=not PREDICTION_LOG_PATH.exists(), index=False
    )


def prediction_summary() -> dict:
    if not PREDICTION_LOG_PATH.exists():
        return {"prediction_count": 0, "average_probability": None,
                "class_distribution": {}, "counts_by_day": {}}
    log = pd.read_csv(PREDICTION_LOG_PATH)
    if log.empty:
        return {"prediction_count": 0, "average_probability": None,
                "class_distribution": {}, "counts_by_day": {}}
    dates = pd.to_datetime(log["timestamp_utc"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")
    return {
        "prediction_count": int(len(log)),
        "average_probability": float(log["probability"].mean()),
        "class_distribution": {str(key): int(value) for key, value in log["prediction"].value_counts().items()},
        "counts_by_day": {str(key): int(value) for key, value in dates.value_counts().sort_index().items()},
    }


if __name__ == "__main__":
    print(json.dumps(prediction_summary(), indent=2))
