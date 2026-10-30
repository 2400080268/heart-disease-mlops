import numpy as np

from src.config import FEATURE_COLUMNS
from src.data_loader import load_dataset
from src.predict import load_model, predict_patient


def test_saved_model_predicts_and_returns_probability():
    dataset = load_dataset()
    model = load_model()
    prediction = model.predict(dataset[FEATURE_COLUMNS].iloc[:3])
    probability = model.predict_proba(dataset[FEATURE_COLUMNS].iloc[:3])[:, 1]
    assert prediction.shape == (3,)
    assert np.isfinite(probability).all()
    assert ((probability >= 0) & (probability <= 1)).all()


def test_prediction_module_returns_expected_fields():
    row = load_dataset()[FEATURE_COLUMNS].iloc[0].to_dict()
    result = predict_patient(row)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["model_version"]
