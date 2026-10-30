"""FastAPI endpoints for model prediction and basic monitoring."""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import MODEL_VERSION
from src.monitoring import log_prediction, prediction_summary
from src.predict import predict_patient


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=1, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(description="0=female, 1=male in this dataset")
    cp: Literal[1, 2, 3, 4] = Field(description="Chest pain type code")
    trestbps: int = Field(ge=50, le=300, description="Resting blood pressure in mm Hg")
    chol: int = Field(ge=50, le=800, description="Serum cholesterol in mg/dL")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar above 120 mg/dL")
    restecg: Literal[0, 1, 2] = Field(description="Resting ECG category")
    thalach: int = Field(ge=40, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina")
    oldpeak: float = Field(ge=0, le=10, description="ST depression value")
    slope: Literal[1, 2, 3] = Field(description="Peak exercise ST segment slope")
    ca: Literal[0, 1, 2, 3] = Field(description="Major vessels colored by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(description="Thalassemia category")


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability: float
    model_version: str
    disclaimer: str


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational model API. Predictions are not medical diagnoses.",
    version=MODEL_VERSION,
)


@app.get("/")
def root():
    return {"name": "Heart Disease Prediction API", "model_version": MODEL_VERSION,
            "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metrics")
def metrics():
    return prediction_summary()


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    values = patient.model_dump()
    try:
        result = predict_patient(values)
        log_prediction(values, result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["disclaimer"] = "Educational use only; this is not a medical diagnosis."
    return result
