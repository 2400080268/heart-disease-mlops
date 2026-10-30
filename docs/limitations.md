# Limitations and responsible use

- The Cleveland subset has only 303 historical records and is not a current, representative clinical population.
- A single stratified holdout split gives a noisy estimate. Cross-validation and external validation would be needed for stronger research evaluation.
- The original target is a clinical label encoded 0–4; binary grouping discards severity distinctions.
- Missing values are imputed; this may mask meaningful missingness patterns.
- ROC-AUC and probabilities from these models do not establish clinical utility, calibration, fairness, or safety.
- The app has no authentication, persistence service, alerting, or production deployment controls. Prediction logging is a local CSV demonstration.
- The historical dataset is not appropriate for making decisions about a real person's care. Seek qualified medical advice for health concerns.

**This educational project is not a medical diagnostic system.**
