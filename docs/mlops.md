# MLOps workflow

1. **Data:** `scripts/download_data.py` retrieves the public processed Cleveland file. `src/data_validation.py` checks schema, usable rows, valid category codes, numeric values, and impossible negative measurements.
2. **Preprocessing:** `src/preprocessing.py` builds a scikit-learn `ColumnTransformer`. Imputers, scaler, and one-hot encoder are fit as part of the model pipeline on training data only.
3. **Training and selection:** `src/train.py` trains Logistic Regression and Random Forest, calculates five held-out metrics, and selects by ROC-AUC.
4. **Tracking:** MLflow stores candidate parameters, dataset metadata, metrics, timestamps, and each fitted pipeline in local storage.
5. **Artifacts:** The selected pipeline is saved using joblib. Reports, plots, metrics, and schema are written under `reports/` and `models/`.
6. **Serving:** FastAPI validates request data, returns a class, probability, model version, and disclaimer. Streamlit calls this API.
7. **Monitoring:** API requests are appended to a local CSV. `/metrics` provides count, average positive-class probability, class distribution, and daily counts. This is an educational summary, not production observability.
8. **Packaging and CI:** Docker Compose runs training, API, Streamlit, and the MLflow UI locally. GitHub Actions installs dependencies, downloads data, runs the pipeline, and executes pytest without secrets or cloud services.
