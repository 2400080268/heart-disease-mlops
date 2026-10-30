"""Dataset checks that fail with an actionable message."""

import pandas as pd

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


def validate_raw_data(frame: pd.DataFrame) -> None:
    required = [*FEATURE_COLUMNS, "target"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame["target"].isna().all():
        raise ValueError("Target column has no usable values.")
    if frame["target"].isna().any():
        raise ValueError("Target column contains missing labels; remove those rows before training.")

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"Feature '{column}' has no numeric values.")
        if (values.dropna() < 0).any():
            raise ValueError(f"Feature '{column}' contains impossible negative values.")

    allowed = {
        "sex": {0, 1}, "cp": {1, 2, 3, 4}, "fbs": {0, 1},
        "restecg": {0, 1, 2}, "exang": {0, 1}, "slope": {1, 2, 3},
        "ca": {0, 1, 2, 3}, "thal": {3, 6, 7},
    }
    for column in CATEGORICAL_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        invalid = sorted(set(values.unique()) - allowed[column])
        if invalid:
            raise ValueError(f"Feature '{column}' has unsupported values: {invalid}")

    target = pd.to_numeric(frame["target"], errors="coerce").dropna()
    invalid_targets = sorted(set(target.unique()) - {0, 1, 2, 3, 4})
    if invalid_targets:
        raise ValueError(f"Target must contain values 0 through 4; found {invalid_targets}")
