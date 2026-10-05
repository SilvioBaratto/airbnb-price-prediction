"""Download the Inside Airbnb dump for Rome and clean it into the committed snapshot.

The upstream ``listings.csv.gz`` has 90 columns, among them host names, photos, profile URLs and
free-text descriptions. The snapshot keeps only what the model reads (``config.SNAPSHOT_COLUMNS``)
plus the listing id, so the committed file carries no host personal data.

The parsing helpers are pure ``pandas`` transforms and are tested on handmade frames; only
:func:`download` and :func:`main` touch the network or the disk.

Usage::

    airbnb fetch-data                     # download if missing, then rebuild the snapshot
    airbnb fetch-data --force-download    # re-download even if the dump is already on disk
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from airbnb import paths
from airbnb.domain import config, geo

# Upstream columns the cleaning step reads; everything else (host data included) is never loaded.
RAW_COLUMNS = [
    "id",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "price",
    "price_quote_checkin_date",
    "price_quote_checkout_date",
    "minimum_nights",
    "availability_365",
    "number_of_reviews",
    "review_scores_rating",
    "host_is_superhost",
]


@dataclass(frozen=True)
class CleanReport:
    """How many listings each cleaning rule dropped, in the order the rules apply."""

    n_raw: int
    n_unpriced: int
    n_long_stay_quote: int
    n_out_of_range: int
    n_kept: int


# --- Parsing helpers (pure) --------------------------------------------------
def parse_price(price: pd.Series) -> pd.Series:
    """Turn Inside Airbnb's ``"$1,234.50"`` strings into floats (``NaN`` when unpriced).

    The ``$`` is a formatting artefact of the dump: the quotes for Rome are in EUR, as the
    ``currency`` field of ``price_quote_raw`` states.
    """
    cleaned = price.astype("string").str.replace(r"[$,]", "", regex=True)
    return cast(pd.Series, pd.to_numeric(cleaned, errors="coerce")).astype(float)


def quote_nights(checkin: pd.Series, checkout: pd.Series) -> pd.Series:
    """Return the number of nights of the stay each price was quoted for."""
    return (pd.to_datetime(checkout) - pd.to_datetime(checkin)).dt.days.astype(float)


def parse_bathrooms(bathrooms: pd.Series, text: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return ``(bathrooms, shared)`` from the numeric column and its free-text twin.

    The numeric column is missing for ~15% of Rome's listings, while ``bathrooms_text``
    (``"1.5 baths"``, ``"1 shared bath"``, ``"Half-bath"``) is almost always there, so the text
    fills the gaps. ``shared`` is 1.0 when the text says the bathroom is shared.
    """
    lowered = text.astype("string").str.lower().fillna("")
    extracted = lowered.str.extract(r"(\d+(?:\.\d+)?)")[0]
    from_text = cast(pd.Series, pd.to_numeric(extracted, errors="coerce"))
    from_text = from_text.where(~(lowered.str.contains("half") & from_text.isna()), 0.5)
    count = cast(pd.Series, pd.to_numeric(bathrooms, errors="coerce")).fillna(from_text)
    shared = lowered.str.contains("shared").astype(float)
    return count.astype(float), shared


def count_amenities(amenities: pd.Series) -> pd.Series:
    """Return how many amenities each listing declares (its JSON list's length, 0 if absent)."""

    def _count(raw: object) -> int:
        if not isinstance(raw, str) or not raw:
            return 0
        try:
            return len(json.loads(raw))
        except json.JSONDecodeError:
            return 0

    return amenities.map(_count).astype(float)


def parse_flag(flag: pd.Series) -> pd.Series:
    """Map Inside Airbnb's ``"t"``/``"f"`` booleans to 1.0/0.0, keeping ``NaN`` for unknown."""
    return flag.map({"t": 1.0, "f": 0.0}).astype(float)


def clean_listings(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleanReport]:
    """Turn the upstream listings into the modeling snapshot.

    Keeps listings with a price quoted for a short stay (``config.MAX_QUOTE_NIGHTS``) inside the
    ``config.PRICE_MIN_EUR``..``PRICE_MAX_EUR`` slice, derives the engineered columns, and
    returns them in ``config.SNAPSHOT_COLUMNS`` order sorted by listing id.

    Args:
        raw: the upstream ``listings.csv.gz`` read with at least ``RAW_COLUMNS``.

    Returns:
        The snapshot frame and a report of how many rows each rule dropped.
    """
    price = parse_price(raw["price"])
    nights = quote_nights(raw["price_quote_checkin_date"], raw["price_quote_checkout_date"])

    priced = price.notna()
    short_stay = nights <= config.MAX_QUOTE_NIGHTS
    in_range = price.between(config.PRICE_MIN_EUR, config.PRICE_MAX_EUR)
    keep = priced & short_stay & in_range

    bathrooms, shared = parse_bathrooms(raw["bathrooms"], raw["bathrooms_text"])
    lat = raw["latitude"].astype(float)
    lon = raw["longitude"].astype(float)
    frame = pd.DataFrame(
        {
            "listing_id": raw["id"].astype("int64"),
            "accommodates": raw["accommodates"].astype(float),
            "bedrooms": pd.to_numeric(raw["bedrooms"], errors="coerce").astype(float),
            "beds": pd.to_numeric(raw["beds"], errors="coerce").astype(float),
            "bathrooms": bathrooms,
            "bathroom_shared": shared,
            "amenities_count": count_amenities(raw["amenities"]),
            "minimum_nights": raw["minimum_nights"].astype(float),
            "availability_365": raw["availability_365"].astype(float),
            "number_of_reviews": raw["number_of_reviews"].astype(float),
            "review_scores_rating": pd.to_numeric(raw["review_scores_rating"], errors="coerce"),
            "host_is_superhost": parse_flag(raw["host_is_superhost"]),
            "latitude": lat,
            "longitude": lon,
            "km_to_center": np.round(geo.km_to_center(lat.to_numpy(), lon.to_numpy()), 3),
            "room_type": raw["room_type"].astype(str),
            "neighbourhood": raw["neighbourhood_cleansed"].astype(str),
            config.TARGET: price.round(2),
        }
    )
    snapshot = (
        frame.loc[keep, config.SNAPSHOT_COLUMNS].sort_values("listing_id").reset_index(drop=True)
    )
    report = CleanReport(
        n_raw=len(raw),
        n_unpriced=int((~priced).sum()),
        n_long_stay_quote=int((priced & ~short_stay).sum()),
        n_out_of_range=int((priced & short_stay & ~in_range).sum()),
        n_kept=len(snapshot),
    )
    return snapshot, report


# --- I/O -----------------------------------------------------------------------
def download(url: str, dest: Path, *, force: bool = False) -> Path:
    """Download ``url`` to ``dest`` unless it is already there; return ``dest``.

    Writes to a ``.part`` file first so an interrupted download never leaves a truncated dump
    that the next run would mistake for a complete one.
    """
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    partial.replace(dest)
    return dest


def build_snapshot(raw_path: Path, out_path: Path) -> CleanReport:
    """Read the upstream dump at ``raw_path``, clean it, and write the snapshot to ``out_path``."""
    raw = pd.read_csv(raw_path, usecols=RAW_COLUMNS, low_memory=False)
    snapshot, report = clean_listings(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(out_path, index=False)
    return report


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the ``fetch-data`` flags."""
    parser = argparse.ArgumentParser(
        prog="airbnb fetch-data",
        description="Download the Inside Airbnb dump for Rome and rebuild the cleaned snapshot.",
    )
    parser.add_argument(
        "--url", default=paths.LISTINGS_URL, help=f"dump URL (default: {paths.LISTINGS_URL})"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=paths.RAW_LISTINGS,
        help="where the dump is cached (default: data/sources/, git-ignored)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.LISTINGS_CSV,
        help="where the cleaned snapshot is written (default: data/raw/listings_rome.csv)",
    )
    parser.add_argument(
        "--force-download", action="store_true", help="download again even if the dump exists"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> CleanReport:
    """Download (if needed) and rebuild the snapshot, printing what each rule dropped."""
    args = parse_cli(argv)
    if args.force_download or not args.raw.exists():
        print(f"Downloading {args.url} ...")
    raw_path = download(args.url, args.raw, force=args.force_download)
    print(f"Cleaning {raw_path} ...")
    report = build_snapshot(raw_path, args.out)
    print(f"  {report.n_raw:>6,} listings in the dump")
    print(f"  {report.n_unpriced:>6,} dropped: no price (not bookable when scraped)")
    print(f"  {report.n_long_stay_quote:>6,} dropped: price quoted for a long stay (discounted)")
    print(
        f"  {report.n_out_of_range:>6,} dropped: outside "
        f"{config.PRICE_MIN_EUR:.0f}-{config.PRICE_MAX_EUR:.0f} EUR per night"
    )
    print(f"  {report.n_kept:>6,} listings kept -> {args.out}")
    return report


if __name__ == "__main__":
    main()
