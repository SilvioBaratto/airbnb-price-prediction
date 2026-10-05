"""Filesystem paths for the project's data files (infrastructure layer).

The single place that knows *where* things live on disk. Kept out of :mod:`airbnb.domain.config`
so the pure domain never needs to know about the filesystem.
"""

from __future__ import annotations

from pathlib import Path

# paths.py lives at airbnb/infrastructure/paths.py, so the project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SOURCES_DIR = DATA_DIR / "sources"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

SNAPSHOT_DATE = "2026-06-20"
LISTINGS_URL = (
    f"https://data.insideairbnb.com/italy/lazio/rome/{SNAPSHOT_DATE}/data/listings.csv.gz"
)

# The upstream dump, downloaded on demand and git-ignored (it carries host personal data).
RAW_LISTINGS = SOURCES_DIR / f"listings_rome_{SNAPSHOT_DATE}.csv.gz"
# The cleaned, host-free snapshot every command reads. Committed.
LISTINGS_CSV = RAW_DIR / "listings_rome.csv"
# Hand-picked spots a host can choose in the simulator. Committed.
PLACES_CSV = SOURCES_DIR / "rome_places.csv"

MODEL_JOBLIB = MODELS_DIR / "price_model.joblib"
