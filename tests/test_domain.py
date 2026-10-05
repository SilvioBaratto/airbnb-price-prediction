"""Pure domain: great-circle distances, the data contract, and value-object validation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from airbnb.domain import config, geo
from airbnb.domain.entities import ListingRequest, Place, PriceQuote

PLACE = Place(1, "Colosseo", "I Centro Storico", 41.8902, 12.4922)


def test_one_degree_of_latitude_is_about_111_km() -> None:
    """A meridian degree is the textbook 111.2 km."""
    assert geo.haversine_km(41.0, 12.0, 42.0, 12.0) == pytest.approx(111.2, abs=0.1)


def test_haversine_is_zero_on_the_same_point_and_symmetric() -> None:
    """Distance is zero to itself and does not depend on direction."""
    assert geo.haversine_km(41.9, 12.5, 41.9, 12.5) == 0.0
    a = geo.haversine_km(41.9, 12.5, 41.7, 12.3)
    b = geo.haversine_km(41.7, 12.3, 41.9, 12.5)
    assert a == pytest.approx(b)


def test_haversine_broadcasts_arrays_against_a_scalar() -> None:
    """Many listings can be measured against one place in a single call."""
    km = geo.haversine_km(np.array([41.9, 42.0]), np.array([12.5, 12.5]), 41.9, 12.5)
    assert isinstance(km, np.ndarray)
    assert km.shape == (2,)
    assert km[0] == 0.0


def test_colosseo_is_about_a_kilometre_from_piazza_venezia() -> None:
    """The centre constant points at Piazza Venezia, not somewhere else."""
    assert geo.km_to_center(PLACE.lat, PLACE.lon) == pytest.approx(1.1, abs=0.1)


def test_feature_contract_is_consistent() -> None:
    """The snapshot is id + features + target, and the target never leaks into the features."""
    assert config.FEATURES == config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    assert len(set(config.FEATURES)) == len(config.FEATURES)
    assert config.SNAPSHOT_COLUMNS[0] == "listing_id"
    assert config.SNAPSHOT_COLUMNS[-1] == config.TARGET
    assert config.TARGET not in config.FEATURES


def test_a_valid_request_is_accepted() -> None:
    """Zero bedrooms is a studio, not an error."""
    request = ListingRequest(PLACE, "Entire home/apt", 2, 0, 1, 1.5)
    assert request.bedrooms == 0


@pytest.mark.parametrize(
    "room_type, guests, bedrooms, beds, bathrooms",
    [
        ("Castle", 2, 1, 1, 1.0),
        ("Entire home/apt", 0, 1, 1, 1.0),
        ("Entire home/apt", 17, 1, 1, 1.0),
        ("Entire home/apt", 2, -1, 1, 1.0),
        ("Entire home/apt", 2, 1, 0, 1.0),
        ("Entire home/apt", 2, 1, 1, 1.25),
    ],
)
def test_an_invalid_request_is_rejected(
    room_type: str, guests: int, bedrooms: int, beds: int, bathrooms: float
) -> None:
    """Anything Airbnb itself would refuse never reaches the model."""
    with pytest.raises(ValueError):
        ListingRequest(PLACE, room_type, guests, bedrooms, beds, bathrooms)


def test_quotes_sort_by_price_only() -> None:
    """``sorted`` lists the cheapest quote first, whatever the room type."""
    cheap = PriceQuote(80.0, "Private room", 60.0, 100.0)
    dear = PriceQuote(150.0, "Entire home/apt", 100.0, 200.0)
    assert sorted([dear, cheap]) == [cheap, dear]


@pytest.mark.parametrize(
    "price, low, high", [(0.0, 0.0, 1.0), (50.0, 60.0, 90.0), (95.0, 60.0, 90.0)]
)
def test_a_quote_outside_its_band_or_free_is_rejected(
    price: float, low: float, high: float
) -> None:
    """A quote must be positive and sit inside its own band."""
    with pytest.raises(ValueError):
        PriceQuote(price, "Private room", low, high)


def test_nan_is_not_a_valid_price() -> None:
    """A NaN from a broken model fails loudly instead of printing "€nan"."""
    with pytest.raises(ValueError):
        PriceQuote(math.nan, "Private room", 1.0, 2.0)
