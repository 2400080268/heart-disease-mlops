from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

SAMPLE_PATIENT = {
    "age": 55, "sex": 1, "cp": 1, "trestbps": 140, "chol": 240,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.2,
    "slope": 1, "ca": 0, "thal": 3,
}


def test_root_and_health():
    assert client.get("/").status_code == 200
    assert client.get("/health").json() == {"status": "healthy"}


def test_predict_and_log():
    response = client.post("/predict", json=SAMPLE_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["probability"] <= 1
    assert "not a medical diagnosis" in body["disclaimer"]
    assert client.get("/metrics").json()["prediction_count"] >= 1


def test_invalid_input_returns_validation_error():
    invalid = dict(SAMPLE_PATIENT)
    invalid["age"] = 200
    assert client.post("/predict", json=invalid).status_code == 422


def test_extra_feature_is_rejected():
    invalid = {**SAMPLE_PATIENT, "patient_name": "not-accepted"}
    assert client.post("/predict", json=invalid).status_code == 422
