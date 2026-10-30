"""Simple Streamlit user interface that calls the FastAPI service."""

import os

import requests
import streamlit as st


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")
st.title("Heart Disease Prediction")
st.write("Enter the clinical attributes below to see the model's educational prediction.")
st.caption("This tool is for educational purposes and is not a medical diagnosis.")

with st.form("patient_form"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sex code", [0, 1], format_func=lambda value: "Female (0)" if value == 0 else "Male (1)")
        cp = st.selectbox("Chest pain type", [1, 2, 3, 4])
        trestbps = st.number_input("Resting blood pressure (mm Hg)", 50, 300, 140)
        chol = st.number_input("Cholesterol (mg/dL)", 50, 800, 240)
        fbs = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1])
        restecg = st.selectbox("Resting ECG category", [0, 1, 2])
    with right:
        thalach = st.number_input("Maximum heart rate", 40, 250, 150)
        exang = st.selectbox("Exercise-induced angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.2, step=0.1)
        slope = st.selectbox("Peak exercise ST slope", [1, 2, 3])
        ca = st.selectbox("Major vessels colored (ca)", [0, 1, 2, 3])
        thal = st.selectbox("Thal category", [3, 6, 7])
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    patient = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    }
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
    try:
        response = requests.post(f"{api_url}/predict", json=patient, timeout=20)
        response.raise_for_status()
        result = response.json()
        st.subheader(result["prediction_label"])
        st.metric("Model probability for the positive class", f"{result['probability']:.1%}")
        st.info(result["disclaimer"])
        st.caption(f"Model version: {result['model_version']}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {api_url}. Start FastAPI and try again. Details: {exc}")
