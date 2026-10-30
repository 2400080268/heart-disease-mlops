"""Train, compare, track, and save candidate classifiers."""

from datetime import datetime, timezone
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_FEATURES, EXPERIMENT_NAME, FEATURE_COLUMNS, MLFLOW_DB_PATH,
    MODEL_DIR, MODEL_PATH, MODEL_VERSION, RANDOM_STATE, SCHEMA_PATH,
    RAW_DATA_PATH,
)
from src.data_loader import load_dataset
from src.evaluate import calculate_metrics, save_evaluation_artifacts
from src.preprocessing import make_preprocessor


def train_pipeline() -> dict:
    data = load_dataset()
    X = data[FEATURE_COLUMNS]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=1,
            )),
        ]),
    }

    tracking_uri = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    results = {}
    fitted = {}

    for name, estimator in candidates.items():
        with mlflow.start_run(run_name=name):
            estimator.fit(X_train, y_train)
            metrics = calculate_metrics(estimator, X_test, y_test)
            fitted[name] = estimator
            results[name] = metrics
            mlflow.log_param("model_type", name)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("dataset", "UCI Cleveland Heart Disease")
            mlflow.log_param("dataset_path", "data/raw/cleveland.data")
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            classifier_params = estimator.named_steps["classifier"].get_params()
            mlflow.log_params({
                f"model_{key}": json.dumps(value, sort_keys=True)
                if isinstance(value, dict) else str(value)
                for key, value in classifier_params.items()
                if value is not None
            })
            mlflow.log_metrics(metrics)
            mlflow.set_tag("training_timestamp_utc", datetime.now(timezone.utc).isoformat())
            mlflow.log_artifact(str(RAW_DATA_PATH), artifact_path="dataset")
            mlflow.sklearn.log_model(
                estimator,
                name="model",
                # These types are part of these locally fitted sklearn pipelines.
                skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
            )

    best_name = max(results, key=lambda model_name: results[model_name]["roc_auc"])
    best_model = fitted[best_name]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(best_model, MODEL_PATH)
    schema = {
        "feature_columns": FEATURE_COLUMNS,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": {"name": "target", "mapping": {"0": "no disease", "1-4": "disease"}},
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "selection_metric": "roc_auc",
        "random_state": RANDOM_STATE,
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    save_evaluation_artifacts(results, best_name, best_model, X_test, y_test)
    summary = {"best_model": best_name, "selection_metric": "roc_auc", "metrics": results}
    print(json.dumps(summary, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    return summary
