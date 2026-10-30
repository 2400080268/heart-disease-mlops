# Heart Disease Prediction with MLOps — Project Report

## Abstract

This project demonstrates a local MLOps workflow for binary classification on the UCI Cleveland Heart Disease dataset. It validates and preprocesses data, compares Logistic Regression and Random Forest, tracks runs with MLflow, saves the selected pipeline, exposes predictions through FastAPI, and provides a Streamlit interface. Actual evaluation results are stored in `reports/metrics/model_metrics.json` and `reports/model_comparison.csv` after running the pipeline; no results are stated here until measured.

## Introduction and problem statement

The project studies whether clinical attributes in a small public dataset can be used by simple classifiers to predict the dataset's binary heart disease label. The task is an educational classification exercise, not a clinical diagnostic objective.

## Objectives

- Validate and preprocess the public dataset without test-set leakage.
- Compare two lightweight classifiers using multiple metrics.
- Track experiments and package the selected pipeline.
- Serve the model locally through an API and a simple web interface.
- Demonstrate basic prediction logging, tests, CI, and containers.

## Literature and background

The UCI Heart Disease dataset is a public benchmark used for classification
experiments. This report focuses on a reproducible student implementation and
does not claim novelty or review clinical literature comprehensively.

## Existing system

A basic workflow may train a model in a notebook and leave its data preparation,
metrics, and serving steps disconnected. This makes it harder to repeat or
demonstrate the same result.

## Proposed system

This project adds scripted validation and training, logged experiments,
reusable model artifacts, a REST API, a web form, local prediction summaries,
automated tests, and container configuration.

## Dataset and preprocessing

The project uses the processed Cleveland subset of the UCI Heart Disease data. Source details and attribution are in `dataset.md`. The original target 0–4 is mapped to 0 for no disease and 1 for values 1–4. Missing numeric values use training-set medians; categorical values use training-set modes and one-hot encoding. Logistic Regression also uses training-set numeric standardization. The split is stratified with random state 42.

## Exploratory data analysis

The notebook inspects row/column counts, descriptive statistics, missingness,
duplicate rows, target class counts, and a numeric correlation heatmap. The
training log reports six missing feature values and zero duplicate records in
the downloaded file. Correlation is descriptive and is not treated as causal.

## Methodology and models

Two sklearn pipelines are trained: Logistic Regression and Random Forest. Each pipeline owns its preprocessing, preventing transformations from being fit on test rows. ROC-AUC is used for model selection; accuracy, precision, recall, and F1 are reported alongside it. Recall is useful to inspect because false negative errors can matter in screening contexts, but no metric makes this model suitable for screening or treatment decisions.

## Model evaluation

The actual holdout results are reported below. The confusion matrix and ROC
curve are generated under `reports/figures/`. Because this is one split from a
small dataset, metrics can change substantially under a different split.

## MLOps implementation

MLflow records one run per model. The selected preprocessing/model pipeline is saved using joblib. FastAPI validates request data, Streamlit sends requests to the API, and predictions are appended to a CSV with a timestamp and model version. Docker Compose and GitHub Actions support local packaging and repeatable checks.

### MLflow

Runs are stored in a local SQLite tracking database and local artifacts. Each
run includes model parameters, random seed, dataset information, a timestamp,
the five evaluation metrics, and the fitted estimator.

### API and frontend

FastAPI provides `/`, `/health`, `/predict`, and `/metrics`; automatic Swagger
and ReDoc pages document requests. Streamlit collects the thirteen fields and
calls the API. Both show an educational-use disclaimer.

### Docker and CI/CD

Docker Compose defines a one-shot trainer, API, Streamlit UI, and MLflow UI.
GitHub Actions downloads the public data, runs the pipeline, and executes tests
on pushes and pull requests. Cloud deployment is not required.

### Monitoring

Each API prediction is appended to a local CSV with timestamp, inputs, class,
probability, and model version. `/metrics` summarizes prediction count,
average probability, class distribution, and counts by day. This is a basic
project demonstration, not production monitoring.

## Results

Run `python scripts/run_pipeline.py` to generate actual metrics and plots. Copy measured values from `reports/metrics/model_metrics.json` into the table below for a submission snapshot.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.8852 | 0.8387 | 0.9286 | 0.8814 | 0.9665 |
| Random Forest | 0.8852 | 0.8182 | 0.9643 | 0.8852 | 0.9513 |

Selected model: Logistic Regression, by the highest ROC-AUC (0.9665) on this run. The evaluation is a single holdout split and should be interpreted cautiously. Exact full-precision values are in the generated metrics JSON.

## Limitations and future scope

The dataset is small and historical, and the models are not clinically validated. Future research could include repeated stratified cross-validation, external datasets, calibration and fairness analysis, stronger data governance, and authenticated production serving. See `limitations.md`.

## Conclusion

The project demonstrates a complete local model lifecycle from public data to tracked training, API inference, interface, monitoring summary, tests, CI, and container packaging. All performance claims must be taken from the generated run artifacts. The application is educational and not for diagnosis.

## References

1. Janosi et al. (1989), UCI Heart Disease dataset, DOI 10.24432/C52P4X.
2. Scikit-learn documentation: https://scikit-learn.org/stable/.
3. MLflow documentation: https://mlflow.org/docs/latest/.
4. FastAPI documentation: https://fastapi.tiangolo.com/.
