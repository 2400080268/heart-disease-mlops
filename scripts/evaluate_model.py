"""Print the metrics written by the training pipeline."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import METRICS_DIR


if __name__ == "__main__":
    path = METRICS_DIR / "model_metrics.json"
    if not path.exists():
        raise SystemExit("Metrics are missing. Run `python scripts/run_pipeline.py` first.")
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
