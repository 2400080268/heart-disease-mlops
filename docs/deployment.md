# Local and Docker deployment

## Local

Run `python scripts/run_pipeline.py`, then start three terminals from the project root:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
streamlit run streamlit_app/app.py
```

## Docker Compose

Install Docker Desktop and start its engine, then run:

```powershell
docker compose up --build
```

Compose builds the image, downloads/trains in a one-shot trainer service, and starts API, Streamlit, and MLflow UI. The project directory is mounted so model, tracking, and prediction-log files are visible locally. Endpoints: Streamlit 8501, FastAPI 8000, MLflow 5000. Stop with `Ctrl+C` and optionally `docker compose down`.

No cloud deployment is required. The project intentionally keeps infrastructure local and suitable for a student demonstration.
