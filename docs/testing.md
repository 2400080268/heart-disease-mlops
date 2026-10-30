# Testing

Install the development requirements and download/train once:

```powershell
python -m pip install -r requirements-dev.txt
python scripts/run_pipeline.py
pytest -q
```

Tests cover dataset shape and target mapping, validation errors, preprocessing with missing values, saved model predictions/probabilities, API health and prediction responses, monitoring log updates, and invalid API input. The CI workflow performs the same data preparation and training before running tests.
