"""A synthetic snapshot whose price genuinely depends on the features.

Tests never read the committed ``data/raw/listings_rome.csv``: they build a few hundred listings
in memory, so the suite is fast, hermetic, and independent of the upstream data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airbnb.domain import config, geo

HOODS = {
    "I Centro Storico": (41.8970, 12.4780),
    "II Parioli/Nomentano": (41.9200, 12.5000),
    "X Ostia/Acilia": (41.7320, 12.2850),
}
ROOM_PREMIUM = {
    "Entire home/apt": 60.0,
    "Private room": 0.0,
    "Hotel room": 30.0,
    "Shared room": -40.0,
}


def make_listings(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Return ``n`` synthetic listings in ``config.SNAPSHOT_COLUMNS`` order."""
    rng = np.random.default_rng(seed)
    hood = rng.choice(list(HOODS), size=n, p=[0.6, 0.25, 0.15])
    lat = np.array([HOODS[h][0] for h in hood]) + rng.normal(0, 0.004, n)
    lon = np.array([HOODS[h][1] for h in hood]) + rng.normal(0, 0.004, n)
    room = rng.choice(list(config.ROOM_TYPES), size=n, p=[0.7, 0.24, 0.04, 0.02])
    guests = rng.integers(1, 7, n).astype(float)
    bedrooms = np.maximum(1, guests // 2)
    bathrooms = np.where(guests > 4, 2.0, 1.0)
    km = geo.km_to_center(lat, lon)
    price = (
        40
        + 25 * guests
        + 35 * bathrooms
        - 3 * km
        + np.array([ROOM_PREMIUM[r] for r in room])
        + rng.normal(0, 15, n)
    )
    frame = pd.DataFrame(
        {
            "listing_id": np.arange(1000, 1000 + n),
            "accommodates": guests,
            "bedrooms": np.where(rng.random(n) < 0.1, np.nan, bedrooms),
            "beds": guests,
            "bathrooms": bathrooms,
            "bathroom_shared": (room == "Shared room").astype(float),
            "amenities_count": rng.integers(5, 60, n).astype(float),
            "minimum_nights": rng.integers(1, 5, n).astype(float),
            "availability_365": rng.integers(0, 366, n).astype(float),
            "number_of_reviews": rng.integers(0, 300, n).astype(float),
            "review_scores_rating": np.where(
                rng.random(n) < 0.1, np.nan, rng.uniform(3.5, 5, n).round(2)
            ),
            "host_is_superhost": rng.integers(0, 2, n).astype(float),
            "latitude": lat,
            "longitude": lon,
            "km_to_center": km.round(3),
            "room_type": room,
            "neighbourhood": hood,
            "price_eur": np.clip(price, config.PRICE_MIN_EUR, config.PRICE_MAX_EUR).round(2),
        }
    )
    return frame[config.SNAPSHOT_COLUMNS]
