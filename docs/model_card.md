# Model card: Heart Disease Prediction

## Model name

Selected model: Logistic Regression (the current run selected it by ROC-AUC). The fitted artifact is `models/best_model.joblib`.

## Intended use

Demonstrating a student MLOps workflow using a historical, public dataset.

## Non-intended use

Diagnosis, screening, triage, treatment, or any real patient-care decision.

## Data and target

Processed UCI Cleveland Heart Disease subset. Thirteen clinical attributes are used. Source target 0 maps to no disease; source targets 1–4 map to disease. Missing values are imputed within training pipelines.

## Evaluation

Current run, one stratified 20% holdout: accuracy 0.8852, precision 0.8387, recall 0.9286, F1 0.8814, ROC-AUC 0.9665. Selection metric is ROC-AUC. Re-run values are written to `reports/metrics/model_metrics.json` and replace this snapshot if the code or data changes.

## Limitations and ethical considerations

The data is small, historical, and not representative of all populations. Class labels are an imperfect proxy for health outcomes. No clinical validity, fairness, calibration, or safety claim is made. Do not infer an individual's health status from the output.

**This is an educational model and not a medical diagnostic system.**
