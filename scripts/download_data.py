"""Download the processed Cleveland dataset from the UCI archive."""

import sys
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import RAW_DATA_PATH

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def main() -> None:
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(DATA_URL, timeout=30) as response:
            content = response.read()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the UCI Cleveland dataset. Check your internet connection "
            f"and try again. Source: {DATA_URL}"
        ) from exc
    if not content or len(content.splitlines()) < 100:
        raise RuntimeError("The downloaded dataset appears empty or incomplete.")
    RAW_DATA_PATH.write_bytes(content)
    print(f"Downloaded {len(content.splitlines())} rows to {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()
