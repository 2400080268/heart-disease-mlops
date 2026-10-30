"""Run data download, validation, training, evaluation, and artifact creation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DATA_PATH
from src.data_loader import load_dataset
from src.train import train_pipeline


def main() -> None:
    if not RAW_DATA_PATH.exists():
        from scripts.download_data import main as download_data
        download_data()
    dataset = load_dataset()
    print(f"Validated {len(dataset)} rows and {len(dataset.columns) - 1} features.")
    train_pipeline()


if __name__ == "__main__":
    main()
