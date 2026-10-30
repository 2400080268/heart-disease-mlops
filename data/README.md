# Data folder

Run `python scripts/download_data.py` to download the processed Cleveland file
from the UCI Machine Learning Repository into `raw/cleveland.data`. The training
pipeline validates it and writes the binary target version to `processed/`.
Raw and generated data files are ignored by Git; see `docs/dataset.md` for the
source, attribution, and column definitions.
