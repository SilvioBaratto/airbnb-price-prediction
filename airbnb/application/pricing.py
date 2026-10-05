"""PricingService — the price-a-listing use case (application layer).

Given the listing a host describes, build its feature row, ask the injected
:class:`~airbnb.application.ports.PricePredictor` for the suggested price and its band, and,
when a :class:`~airbnb.application.ports.ListingRepository` is wired in, fetch the real listings
nearby that look like it. Depends only on the domain and the ports — no I/O here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from airbnb.application.ports import ListingRepository, PricePredictor
from airbnb.domain import config, geo
from airbnb.domain.entities import Comparable, ListingRequest, PriceQuote

DEFAULT_COMPARABLES = 5


class PricingService:
    """Turn a listing request into a suggested nightly price and its nearby comparables."""

    def __init__(
        self, predictor: PricePredictor, listings: ListingRepository | None = None
    ) -> None:
        self._predictor = predictor
        self._listings = listings

    def features(self, request: ListingRequest) -> pd.DataFrame:
        """Return the one-row model input for ``request``, in ``config.FEATURES`` order.

        Columns a host does not state (reviews, rating, availability, amenities, ...) are left
        ``NaN``. The fitted pipeline imputes them with the training medians, so the quote is the
        price of a *typical* listing with this size and location, not of a brand-new one with
        zero reviews.
        """
        place = request.place
        row: dict[str, object] = dict.fromkeys(config.FEATURES, np.nan)
        row.update(
            accommodates=float(request.guests),
            bedrooms=float(request.bedrooms),
            beds=float(request.beds),
            bathrooms=float(request.bathrooms),
            latitude=place.lat,
            longitude=place.lon,
            km_to_center=geo.km_to_center(place.lat, place.lon),
            room_type=request.room_type,
            neighbourhood=place.neighbourhood,
        )
        return pd.DataFrame([row], columns=config.FEATURES)

    def quote(self, request: ListingRequest) -> PriceQuote:
        """Return the suggested nightly price for ``request`` and the band around it."""
        frame = self.features(request)
        price = float(self._predictor.predict(frame)[0])
        low, high = self._predictor.predict_band(frame)
        return PriceQuote(price, request.room_type, float(low[0]), float(high[0]))

    def comparables(
        self, request: ListingRequest, k: int = DEFAULT_COMPARABLES
    ) -> list[Comparable]:
        """Return up to ``k`` real listings like ``request``, nearest first (none if unwired)."""
        if self._listings is None:
            return []
        return self._listings.comparables(request, k)
