"""The pricing use case, driven through fake ports."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from airbnb.application.pricing import PricingService
from airbnb.domain import config, geo
from airbnb.domain.entities import Comparable, ListingRequest, Place

PLACE = Place(7, "Trastevere", "I Centro Storico", 41.8894, 12.4700)
REQUEST = ListingRequest(PLACE, "Private room", 2, 1, 1, 1.0)


class FakePredictor:
    """Answers 100 EUR in a 80-130 band, and remembers the frame it was asked about."""

    def __init__(self) -> None:
        self.frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return 100 EUR per row."""
        self.frames.append(frame)
        return np.full(len(frame), 100.0)

    def predict_band(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return an 80-130 EUR band per row."""
        return np.full(len(frame), 80.0), np.full(len(frame), 130.0)


class FakeListings:
    """Returns ``k`` copies of one comparable, recording the request it was asked about."""

    def __init__(self) -> None:
        self.requests: list[ListingRequest] = []

    def comparables(self, request: ListingRequest, k: int) -> list[Comparable]:
        """Return ``k`` identical comparables."""
        self.requests.append(request)
        return [Comparable(1, 0.1, request.room_type, request.guests, 1.0, 1.0, 90.0)] * k


def test_features_fill_what_the_host_states_and_leave_the_rest_blank() -> None:
    """Stated fields are set, derived distance is computed, everything else is NaN to impute."""
    row = PricingService(FakePredictor()).features(REQUEST)
    assert list(row.columns) == config.FEATURES and len(row) == 1
    first = row.iloc[0]
    assert first["accommodates"] == 2 and first["bathrooms"] == 1.0
    assert first["room_type"] == "Private room"
    assert first["neighbourhood"] == "I Centro Storico"
    assert first["km_to_center"] == pytest.approx(geo.km_to_center(PLACE.lat, PLACE.lon))
    for blank in ("number_of_reviews", "review_scores_rating", "amenities_count"):
        assert np.isnan(first[blank])


def test_quote_wraps_the_predictor_answer() -> None:
    """The quote carries the predicted price, its band and the requested room type."""
    predictor = FakePredictor()
    quote = PricingService(predictor).quote(REQUEST)
    assert (quote.price_eur, quote.low_eur, quote.high_eur) == (100.0, 80.0, 130.0)
    assert quote.room_type == "Private room"
    assert len(predictor.frames) == 1


def test_comparables_come_from_the_repository_when_wired() -> None:
    """Comparables are delegated with the request and the requested count."""
    repo = FakeListings()
    found = PricingService(FakePredictor(), repo).comparables(REQUEST, k=3)
    assert len(found) == 3 and repo.requests == [REQUEST]


def test_comparables_are_empty_without_a_repository() -> None:
    """The service still prices a listing when no listings store is wired."""
    assert PricingService(FakePredictor()).comparables(REQUEST) == []
