# Architecture

```mermaid
flowchart TD
  A[UCI Cleveland raw file] --> B[Column and value validation]
  B --> C[Binary target mapping]
  C --> D[Stratified train/test split]
  D --> E[Imputation, scaling, one-hot encoding]
  E --> F[Logistic Regression]
  E --> G[Random Forest]
  F --> H[Held-out metrics]
  G --> H
  F --> I[MLflow run]
  G --> I
  H --> J[Select by ROC-AUC]
  J --> K[Joblib model + schema]
  K --> L[FastAPI prediction endpoint]
  L --> M[Streamlit form]
  L --> N[Prediction CSV + /metrics]
  O[GitHub Actions] --> P[Pipeline and pytest]
```

The saved model is the complete fitted scikit-learn pipeline, including preprocessing. The API does not refit transformations at request time. Both models use the same fixed, stratified split for a fair comparison. MLflow records an independent run for each candidate. ROC-AUC is the selection metric; the full metric set is retained for interpretation.
