# API guide

Start the pipeline first, then run `uvicorn api.main:app --reload` from the project root. Interactive OpenAPI documentation is at `/docs`; ReDoc is at `/redoc`.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | API name, version, docs path |
| GET | `/health` | Liveness check |
| POST | `/predict` | Validated prediction; request has all 13 features |
| GET | `/metrics` | Aggregate counts from the local prediction log |

Example:

```powershell
$patient = @{
  age=55; sex=1; cp=1; trestbps=140; chol=240; fbs=0; restecg=1
  thalach=150; exang=0; oldpeak=1.2; slope=1; ca=0; thal=3
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
  -ContentType 'application/json' -Body $patient
```

Input categories and plausible ranges are validated. Invalid or missing fields receive HTTP 422. The API returns an educational prediction label and model probability, plus an explicit disclaimer. Probability is not a calibrated clinical risk estimate.
