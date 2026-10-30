"""Loading and normalization for the UCI Cleveland heart disease data."""

from pathlib import Path

import pandas as pd

from src.config import FEATURE_COLUMNS, PROCESSED_DATA_PATH, RAW_COLUMNS, RAW_DATA_PATH
from src.data_validation import validate_raw_data


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the official Cleveland text file, where '?' denotes missing data."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` first."
        )
    try:
        frame = pd.read_csv(path, header=None, names=RAW_COLUMNS, na_values="?")
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read dataset at {path}: {exc}") from exc

    if len(frame.columns) != len(RAW_COLUMNS):
        raise ValueError(
            f"Expected {len(RAW_COLUMNS)} columns in Cleveland data; got {len(frame.columns)}."
        )
    for column in RAW_COLUMNS:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & converted.isna()
        if invalid.any():
            examples = frame.loc[invalid, column].astype(str).unique()[:3].tolist()
            raise ValueError(f"Column '{column}' contains non-numeric values: {examples}")
        frame[column] = converted
    validate_raw_data(frame)
    return frame


def prepare_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the 13 published features and convert target 0..4 to 0/1."""
    validate_raw_data(frame)
    missing_count = int(frame[FEATURE_COLUMNS].isna().sum().sum())
    duplicate_count = int(frame.duplicated().sum())
    print(f"Data checks: {missing_count} missing feature values; {duplicate_count} duplicate rows.")
    prepared = frame.loc[:, [*FEATURE_COLUMNS, "target"]].copy()
    prepared["target"] = (prepared["target"] > 0).astype(int)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(PROCESSED_DATA_PATH, index=False)
    return prepared


def load_dataset() -> pd.DataFrame:
    return prepare_dataset(load_raw_data())
