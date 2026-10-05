"""CSV-backed repositories implementing the application ports (infrastructure layer).

:class:`CsvPlaceRepository` reads the hand-picked spots in ``data/sources/rome_places.csv`` and
labels each with the municipio its surrounding listings belong to. The label comes from the data,
not from a hand-written table, so the place speaks the same ``neighbourhood`` vocabulary the
model was trained on. :class:`CsvListingRepository` answers "what do listings like mine nearby
actually ask?" from the cleaned snapshot.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from airbnb import paths
from airbnb.domain import geo
from airbnb.domain.entities import Comparable, ListingRequest, Place

# Enough listings to outvote a stray one across a municipio border, few enough (~250 m in the
# centre) to stay local.
NEIGHBOUR_VOTES = 25


def _fold(text: str) -> str:
    """Lower-case ``text`` and strip its accents, so ``"citta"`` matches ``"Città"``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def nearest_neighbourhood(listings: pd.DataFrame, lat: float, lon: float) -> str:
    """Return the most common ``neighbourhood`` among the listings closest to ``(lat, lon)``."""
    km = geo.haversine_km(
        listings["latitude"].to_numpy(dtype=float),
        listings["longitude"].to_numpy(dtype=float),
        lat,
        lon,
    )
    closest = np.argsort(km)[:NEIGHBOUR_VOTES]
    return str(listings["neighbourhood"].iloc[closest].mode().iloc[0])


class CsvPlaceRepository:
    """A :class:`~airbnb.application.ports.PlaceRepository` over the places and listings CSVs."""

    def __init__(
        self,
        places_path: Path | str | None = None,
        listings_path: Path | str | None = None,
        *,
        listings: pd.DataFrame | None = None,
    ) -> None:
        places = pd.read_csv(places_path or paths.PLACES_CSV)
        if listings is None:
            listings = pd.read_csv(
                listings_path or paths.LISTINGS_CSV,
                usecols=["latitude", "longitude", "neighbourhood"],
            )
        self._places = [
            Place(
                place_id=int(row.place_id),
                name=str(row.name),
                neighbourhood=nearest_neighbourhood(listings, float(row.lat), float(row.lon)),
                lat=float(row.lat),
                lon=float(row.lon),
            )
            for row in places.itertuples(index=False)
        ]
        self._by_id = {p.place_id: p for p in self._places}

    def all(self) -> list[Place]:
        """Return every place, in the CSV's order."""
        return list(self._places)

    def get(self, place_id: int) -> Place:
        """Return the place with this id.

        Raises:
            KeyError: if no place has this id.
        """
        return self._by_id[place_id]

    def search(self, query: str) -> list[Place]:
        """Return the places whose name contains ``query``, ignoring case and accents.

        An empty or blank query matches nothing rather than everything.
        """
        needle = _fold(query.strip())
        if not needle:
            return []
        return [p for p in self._places if needle in _fold(p.name)]


_LISTING_COLUMNS = [
    "listing_id",
    "latitude",
    "longitude",
    "room_type",
    "accommodates",
    "bedrooms",
    "bathrooms",
    "price_eur",
]


def _optional(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


class CsvListingRepository:
    """A :class:`~airbnb.application.ports.ListingRepository` over the cleaned snapshot."""

    def __init__(
        self, listings_path: Path | str | None = None, *, listings: pd.DataFrame | None = None
    ) -> None:
        if listings is None:
            listings = pd.read_csv(listings_path or paths.LISTINGS_CSV, usecols=_LISTING_COLUMNS)
        self._listings = listings[_LISTING_COLUMNS].reset_index(drop=True)

    def comparables(self, request: ListingRequest, k: int) -> list[Comparable]:
        """Return up to ``k`` listings with the request's room type and guests, nearest first.

        When no listing in the city has exactly that room type and guest count (a shared room
        for eight, say), any guest count is accepted rather than showing nothing.
        """
        same_type = self._listings[self._listings["room_type"] == request.room_type]
        exact = same_type[same_type["accommodates"] == request.guests]
        pool = exact if len(exact) else same_type
        km = geo.haversine_km(
            pool["latitude"].to_numpy(dtype=float),
            pool["longitude"].to_numpy(dtype=float),
            request.place.lat,
            request.place.lon,
        )
        nearest = np.argsort(km, kind="stable")[:k]
        return [
            Comparable(
                listing_id=int(row.listing_id),
                km=float(km[i]),
                room_type=str(row.room_type),
                guests=int(row.accommodates),
                bedrooms=_optional(row.bedrooms),
                bathrooms=_optional(row.bathrooms),
                price_eur=float(row.price_eur),
            )
            for i, row in zip(nearest, pool.iloc[nearest].itertuples(index=False))
        ]
