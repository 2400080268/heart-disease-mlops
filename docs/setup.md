# Setup guide

## Requirements

- Python 3.11 (64-bit recommended)
- Internet access for the first UCI dataset download and package install
- Docker Desktop only if running the optional container setup

## Windows PowerShell

From the project root:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/download_data.py
python scripts/run_pipeline.py
```

If activation is blocked for the current terminal, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

## macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python scripts/download_data.py
python scripts/run_pipeline.py
```

The pipeline downloads the data automatically if it is absent. To retrain after changing code, run the pipeline command again. Model and report outputs are recreated. Local MLflow history is retained in the tracking store.
