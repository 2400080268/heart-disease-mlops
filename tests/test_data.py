import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.data_loader import load_dataset
from src.data_validation import validate_raw_data
from src.preprocessing import make_preprocessor


def test_dataset_loads_and_target_is_binary():
    dataset = load_dataset()
    assert len(dataset) > 100
    assert set(FEATURE_COLUMNS).issubset(dataset.columns)
    assert set(dataset["target"].unique()) == {0, 1}


def test_validation_rejects_missing_required_column():
    dataset = load_dataset().drop(columns="age")
    with pytest.raises(ValueError, match="missing required columns"):
        validate_raw_data(dataset)


def test_validation_rejects_invalid_categorical_code():
    dataset = load_dataset()
    dataset.loc[dataset.index[0], "cp"] = 99
    with pytest.raises(ValueError, match="unsupported values"):
        validate_raw_data(dataset)


def test_preprocessing_handles_missing_values():
    dataset = load_dataset()
    X = dataset[FEATURE_COLUMNS].iloc[:10].copy()
    X.loc[X.index[0], "chol"] = float("nan")
    transformed = make_preprocessor(scale_numeric=True).fit_transform(X)
    assert transformed.shape[0] == 10
    assert pd.notna(transformed).all()
