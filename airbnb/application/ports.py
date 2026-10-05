"""Application ports — the interfaces the pricing use case depends on (dependency inversion).

Each port is a ``@runtime_checkable`` :class:`typing.Protocol`, so the adapters in
:mod:`airbnb.infrastructure` (and test fakes) satisfy them structurally without importing this
module. The composition root (:mod:`airbnb.cli.app`) wires in the real implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from airbnb.domain.entities import Comparable, ListingRequest, Place


@runtime_checkable
class PlaceRepository(Protocol):
    """Read access to the spots a host can pick."""

    def all(self) -> list[Place]:
        """Return every place, in catalog order."""
        ...

    def get(self, place_id: int) -> Place:
        """Return the place with this id.

        Raises:
            KeyError: if no place has this id.
        """
        ...

    def search(self, query: str) -> list[Place]:
        """Return the places whose name contains ``query``, ignoring case and accents."""
        ...


@runtime_checkable
class ListingRepository(Protocol):
    """Read access to the real listings, to show a host what similar places ask."""

    def comparables(self, request: ListingRequest, k: int) -> list[Comparable]:
        """Return up to ``k`` listings with the request's room type and guests, nearest first."""
        ...


@runtime_checkable
class PricePredictor(Protocol):
    """A fitted price model over frames shaped like ``config.FEATURES``."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the suggested nightly price (EUR) for each row."""
        ...

    def predict_band(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(low, high)``: the nightly prices most similar listings fall between."""
        ...
