# Viva questions and concise answers

## Machine learning

1. **What is classification?** Predicting a discrete class label from input features.
2. **What is logistic regression?** A linear classification model that estimates class probabilities through a logistic function.
3. **What is a random forest?** An ensemble of decision trees whose predictions are combined to reduce variance.
4. **Why compare these models?** They are lightweight, explainable starting points with different modeling assumptions.
5. **What is a train/test split?** Fit on one subset and estimate performance on held-out examples.
6. **Why stratify the split?** It approximately preserves class proportions in both subsets.
7. **What is overfitting?** Learning noise or training-specific patterns that do not generalize.
8. **What is accuracy?** The fraction of all predictions that are correct.
9. **What is precision?** Of positive predictions, the fraction that are true positives.
10. **What is recall?** Of actual positive cases, the fraction detected by the model.
11. **What is F1?** The harmonic mean of precision and recall.
12. **What is ROC-AUC?** A threshold-independent summary of ranking ability across false-positive rates.
13. **What does a confusion matrix show?** Counts of true/false positives and true/false negatives.
14. **Why not use accuracy alone?** It can hide poor performance on one class and ignores error trade-offs.
15. **What is data leakage?** Information from evaluation data influencing training or preprocessing.
16. **How is leakage avoided here?** Preprocessing is inside sklearn pipelines fitted only on training rows.
17. **Why impute missing values?** Many estimators require complete numeric matrices; imputation keeps usable rows.
18. **Why one-hot encode categories?** It represents categories without implying an ordered numeric distance.
19. **Why scale Logistic Regression inputs?** Scaling helps optimization and balances numeric feature magnitudes.
20. **What does the target mapping mean?** Original 0 becomes class 0; original 1–4 become class 1.

## MLOps and serving

21. **What is MLOps?** Practices that make ML development, deployment, and monitoring repeatable.
22. **Why is MLOps useful?** It records how a model was built and supports reliable updates and serving.
23. **What is MLflow used for here?** Recording model parameters, metrics, artifacts, and separate runs.
24. **What is experiment tracking?** Keeping a structured record of training configurations and outcomes.
25. **What is model versioning?** Identifying model artifacts and the code/data context used to build them.
26. **What is CI?** Automated checks run when code changes or a pull request is opened.
27. **Why Docker?** It packages the application and dependencies in a repeatable environment.
28. **Why FastAPI?** It provides typed request validation and automatic interactive API docs.
29. **Why Streamlit?** It creates a small demo interface in Python with little frontend code.
30. **What is model deployment?** Making a trained model available to an application or user.
31. **What is monitoring in this project?** A local CSV log and summaries of prediction counts and scores.
32. **Is this production observability?** No; it is a basic classroom demonstration.
33. **How does the API work?** It validates a JSON patient record, loads the saved pipeline, and returns prediction details.
34. **How does Streamlit reach the model?** It submits the entered values to FastAPI's `/predict` endpoint.
35. **Where is the trained model stored?** `models/best_model.joblib`.

## Project details and limitations

36. **Why use the Cleveland dataset?** It is a documented public dataset commonly used for educational classification.
37. **Why binary classification?** The exercise groups disease severity labels into presence versus absence.
38. **How is the best model selected?** By the highest ROC-AUC on the fixed held-out test set.
39. **Why include recall?** It shows the share of positive cases found; false negatives deserve attention in health-related examples.
40. **Where are metrics saved?** In `reports/metrics/model_metrics.json` and `reports/model_comparison.csv`.
41. **What is the model probability?** The estimator's positive-class score, not a validated clinical risk estimate.
42. **What happens to API predictions?** Inputs, prediction, score, version, and timestamp are appended to a local CSV.
43. **What does `/health` do?** It reports that the API process is responding.
44. **Why use a random seed?** It makes the split and stochastic model training repeatable.
45. **What is the main dataset limitation?** It is small, historical, and not representative of all populations.
46. **Can this app diagnose heart disease?** No. It is an educational example, not a medical diagnostic system.
