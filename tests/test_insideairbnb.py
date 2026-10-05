"""Getting the data: parsing Inside Airbnb's formats and cleaning the dump into the snapshot."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airbnb.datasource import insideairbnb as iab
from airbnb.domain import config


def _raw(**overrides: list) -> pd.DataFrame:
    """Three upstream-shaped rows, with host personal data the snapshot must not carry."""
    rows = {
        "id": [3, 1, 2],
        "listing_url": ["u3", "u1", "u2"],
        "host_name": ["Anna", "Bruno", "Carla"],
        "description": ["d3", "d1", "d2"],
        "neighbourhood_cleansed": ["I Centro Storico"] * 3,
        "latitude": [41.89, 41.90, 41.91],
        "longitude": [12.48, 12.49, 12.50],
        "room_type": ["Entire home/apt", "Private room", "Entire home/apt"],
        "accommodates": [4, 2, 3],
        "bathrooms": [np.nan, 1.0, np.nan],
        "bathrooms_text": ["1.5 baths", "1 shared bath", "Half-bath"],
        "bedrooms": [2, 1, np.nan],
        "beds": [2, 1, 1],
        "amenities": ['["Wifi", "Kitchen"]', "[]", None],
        "price": ["$1,120.00", "$59.52", "$80.00"],
        "price_quote_checkin_date": ["2026-06-21"] * 3,
        "price_quote_checkout_date": ["2026-06-23"] * 3,
        "minimum_nights": [2, 1, 3],
        "availability_365": [100, 200, 300],
        "number_of_reviews": [10, 0, 5],
        "review_scores_rating": [4.8, np.nan, 4.5],
        "host_is_superhost": ["t", "f", np.nan],
    }
    rows.update(overrides)
    return pd.DataFrame(rows)


def test_parse_price_strips_dollar_sign_and_thousands_separator() -> None:
    """``"$1,120.00"`` is 1120 EUR, and an unpriced listing stays NaN rather than 0."""
    parsed = iab.parse_price(pd.Series(["$1,120.00", "$59.52", np.nan]))
    assert parsed.iloc[0] == 1120.0
    assert parsed.iloc[1] == 59.52
    assert np.isnan(parsed.iloc[2])


def test_quote_nights_counts_the_nights_of_the_quoted_stay() -> None:
    """A 21 June to 22 July quote is a 31-night stay."""
    nights = iab.quote_nights(pd.Series(["2026-06-21"]), pd.Series(["2026-07-22"]))
    assert nights.iloc[0] == 31


def test_parse_bathrooms_fills_gaps_from_the_text_and_flags_shared() -> None:
    """The text twin fills the numeric gaps, halves included, and marks shared bathrooms."""
    count, shared = iab.parse_bathrooms(
        pd.Series([np.nan, 1.0, np.nan, 2.0]),
        pd.Series(["1.5 baths", "1 shared bath", "Half-bath", None]),
    )
    assert count.tolist() == [1.5, 1.0, 0.5, 2.0]
    assert shared.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_count_amenities_tolerates_missing_and_malformed_lists() -> None:
    """A missing or broken amenities field counts as none instead of crashing the clean."""
    counts = iab.count_amenities(pd.Series(['["Wifi", "Kitchen"]', "[]", None, "not json"]))
    assert counts.tolist() == [2.0, 0.0, 0.0, 0.0]


def test_parse_flag_keeps_unknown_as_nan() -> None:
    """An unknown superhost status is imputed later, not silently read as "no"."""
    flags = iab.parse_flag(pd.Series(["t", "f", np.nan]))
    assert flags.iloc[0] == 1.0 and flags.iloc[1] == 0.0
    assert np.isnan(flags.iloc[2])


def test_clean_listings_keeps_the_contract_columns_and_no_host_data() -> None:
    """The snapshot has exactly the contract columns, sorted by id, with derived fields."""
    snapshot, report = iab.clean_listings(_raw())
    assert list(snapshot.columns) == config.SNAPSHOT_COLUMNS
    assert "host_name" not in snapshot.columns and "description" not in snapshot.columns
    assert snapshot["listing_id"].tolist() == [1, 2]  # 1120 EUR is above the range
    assert snapshot["bathroom_shared"].tolist() == [1.0, 0.0]
    assert snapshot["amenities_count"].tolist() == [0.0, 0.0]
    assert (snapshot["km_to_center"] > 0).all()
    assert report.n_kept == 2 and report.n_out_of_range == 1


def test_clean_listings_drops_unpriced_and_long_stay_quotes() -> None:
    """Each rule drops its own rows, and the report accounts for every raw row."""
    raw = _raw(
        price=[np.nan, "$59.52", "$80.00"],
        price_quote_checkout_date=["2026-06-23", "2026-07-22", "2026-06-23"],
    )
    snapshot, report = iab.clean_listings(raw)
    assert snapshot["listing_id"].tolist() == [2]
    assert (report.n_unpriced, report.n_long_stay_quote, report.n_out_of_range) == (1, 1, 0)
    assert report.n_raw == report.n_unpriced + report.n_long_stay_quote + report.n_kept


def test_build_snapshot_reads_a_gzipped_dump_and_writes_the_csv(tmp_path: Path) -> None:
    """The file seam reads only the needed columns from the .gz and writes the snapshot."""
    raw_path = tmp_path / "listings.csv.gz"
    with gzip.open(raw_path, "wt") as fh:
        _raw().to_csv(fh, index=False)
    out = tmp_path / "out" / "snapshot.csv"
    report = iab.build_snapshot(raw_path, out)
    assert report.n_kept == 2
    assert list(pd.read_csv(out).columns) == config.SNAPSHOT_COLUMNS


def test_download_skips_an_existing_file_and_refetches_when_forced(tmp_path: Path) -> None:
    """No network call when the dump is cached, and a forced fetch replaces it atomically."""
    source = tmp_path / "upstream.csv.gz"
    source.write_bytes(b"fresh")
    dest = tmp_path / "cache" / "dump.csv.gz"
    dest.parent.mkdir()
    dest.write_bytes(b"cached")

    assert iab.download("file:///nonexistent", dest).read_bytes() == b"cached"
    assert iab.download(source.as_uri(), dest, force=True).read_bytes() == b"fresh"
    assert not dest.with_suffix(".gz.part").exists()


def test_main_rebuilds_the_snapshot_from_a_cached_dump(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``airbnb fetch-data`` with the dump on disk never downloads, and reports each rule."""
    raw_path = tmp_path / "dump.csv.gz"
    with gzip.open(raw_path, "wt") as fh:
        _raw().to_csv(fh, index=False)
    out = tmp_path / "snapshot.csv"
    report = iab.main(["--raw", str(raw_path), "--out", str(out), "--url", "file:///nope"])
    assert report.n_kept == 2 and out.exists()
    assert "Downloading" not in capsys.readouterr().out
