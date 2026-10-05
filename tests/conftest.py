"""Shared fixtures over the synthetic snapshot built by :mod:`tests.factories`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airbnb.modeling import pipeline
from tests.factories import HOODS, make_listings


@pytest.fixture
def listings() -> pd.DataFrame:
    """The 400-listing synthetic snapshot."""
    return make_listings()


@pytest.fixture
def split(listings: pd.DataFrame) -> pipeline.Split:
    """The fixed 80/20 split of the synthetic snapshot."""
    return pipeline.make_split(listings)


@pytest.fixture
def snapshot_csv(tmp_path: Path, listings: pd.DataFrame) -> Path:
    """The synthetic snapshot written to a CSV, for the file-reading seams."""
    path = tmp_path / "listings.csv"
    listings.to_csv(path, index=False)
    return path


@pytest.fixture
def places_csv(tmp_path: Path) -> Path:
    """Three places, one per synthetic neighbourhood; the first name carries an accent."""
    path = tmp_path / "places.csv"
    pd.DataFrame(
        {
            "place_id": [1, 2, 3],
            "name": ["Città Vecchia", "Parioli", "Ostia"],
            "lat": [HOODS["I Centro Storico"][0], HOODS["II Parioli/Nomentano"][0], 41.7320],
            "lon": [HOODS["I Centro Storico"][1], HOODS["II Parioli/Nomentano"][1], 12.2850],
        }
    ).to_csv(path, index=False)
    return path
