"""Project paths and shared model configuration."""

from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "cleveland.data"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "heart_disease.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
REPORTS_DIR = ROOT_DIR / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"
PREDICTION_LOG_PATH = METRICS_DIR / "prediction_log.csv"
MLFLOW_DB_PATH = ROOT_DIR / "mlflow.db"
RANDOM_STATE = 42
EXPERIMENT_NAME = "heart-disease-prediction"
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

RAW_COLUMNS = [*FEATURE_COLUMNS, "target"]
