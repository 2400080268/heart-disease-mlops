# Presentation outline (15 slides)

1. **Title:** Heart Disease Prediction with MLOps. Introduce the team/course.
2. **Problem statement:** Demonstrate a repeatable ML lifecycle on public data.
3. **Motivation:** A trained model also needs versioning, serving, and checks.
4. **Objectives:** Data validation, model comparison, tracking, API, UI, CI.
5. **Dataset:** UCI Cleveland subset, 303 records, 13 attributes; target mapping.
6. **Architecture:** Use the Mermaid diagram in `architecture.mmd`.
7. **Preprocessing:** Missing values, one-hot categories, scaling, stratified split.
8. **Models:** Logistic Regression and Random Forest; explain simplicity.
9. **Evaluation:** Insert actual metrics and generated confusion/ROC plots.
10. **MLflow:** Show the experiment page and two candidate runs.
11. **API and UI:** Demonstrate `/docs`, `/predict`, and the Streamlit form.
12. **Monitoring:** Show local prediction log and `/metrics` summary.
13. **Docker and CI:** Show Compose services and passing GitHub workflow.
14. **Limitations:** Historical small dataset, holdout uncertainty, no clinical use.
15. **Conclusion and future scope:** Summarize lifecycle; propose validation and calibration research.

## Speaker note

Use only metrics from the generated `model_metrics.json`. State clearly that the output is an educational model prediction, not medical advice.
