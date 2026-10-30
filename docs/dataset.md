# Dataset and data decisions

## Source

- Dataset: UCI Machine Learning Repository, Heart Disease (processed Cleveland subset).
- Record: https://archive.ics.uci.edu/dataset/45/heart%2Bdisease
- Downloaded file: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
- Citation: Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989), *Heart Disease*, UCI Machine Learning Repository, DOI [10.24432/C52P4X](https://doi.org/10.24432/C52P4X).
- UCI lists the dataset under Creative Commons Attribution 4.0 (CC BY 4.0); credit UCI and the dataset authors when redistributing it.

The processed Cleveland subset contains 303 records and 13 predictor columns. The raw data file is comma-separated without a header. `?` marks missing values. The download script saves the original file in `data/raw/`.

## Features

| Column | Meaning | Treatment |
| --- | --- | --- |
| age | Age in years | Numeric, median imputation |
| sex | Dataset code: 1 male, 0 female | Categorical, most-frequent imputation, one-hot encoded |
| cp | Chest pain type (1–4) | Categorical, one-hot encoded |
| trestbps | Resting blood pressure (mm Hg) | Numeric, median imputation and scaling for Logistic Regression |
| chol | Serum cholesterol (mg/dL) | Numeric, median imputation and scaling for Logistic Regression |
| fbs | Fasting blood sugar >120 mg/dL (1 true, 0 false) | Categorical, one-hot encoded |
| restecg | Resting ECG category (0–2) | Categorical, one-hot encoded |
| thalach | Maximum heart rate achieved | Numeric, median imputation and scaling for Logistic Regression |
| exang | Exercise-induced angina (1 yes, 0 no) | Categorical, one-hot encoded |
| oldpeak | ST depression value | Numeric, median imputation and scaling for Logistic Regression |
| slope | Peak exercise ST segment slope (1–3) | Categorical, one-hot encoded |
| ca | Major vessels colored by fluoroscopy (0–3) | Categorical, one-hot encoded |
| thal | Thal category (3, 6, 7) | Categorical, one-hot encoded |

## Target and preprocessing

The source `num` value is 0–4. The project maps source 0 to binary target 0 (no disease) and source 1, 2, 3, or 4 to target 1 (heart disease present). Missing feature values are imputed inside scikit-learn pipelines: median for numeric fields and most frequent for categorical fields. Numeric scaling is used for Logistic Regression. The split is stratified and performed before fitting preprocessing, preventing test data from affecting imputation or scaling. Duplicate counts and missing counts are reported at data preparation.

The dataset is historical and small. Its labels, features, and population do not support clinical use or generalization claims.
