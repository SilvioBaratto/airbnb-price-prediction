"""Great-circle geometry on the WGS84 sphere (pure domain, no I/O)."""

from __future__ import annotations

from typing import overload

import numpy as np

from airbnb.domain import config

EARTH_RADIUS_KM = 6371.0088


@overload
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float: ...
@overload
def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: float | np.ndarray, lon2: float | np.ndarray
) -> np.ndarray: ...
def haversine_km(lat1, lon1, lat2, lon2):
    """Return the great-circle distance in kilometres between two points (degrees).

    Broadcasts like numpy: pass arrays for one end and scalars for the other to measure many
    points against a single reference. A scalar input returns a plain ``float``.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlmb = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    return float(km) if np.ndim(km) == 0 else km


@overload
def km_to_center(lat: float, lon: float) -> float: ...
@overload
def km_to_center(lat: np.ndarray, lon: np.ndarray) -> np.ndarray: ...
def km_to_center(lat, lon):
    """Return the distance in kilometres from the city centre (``config.CITY_CENTER_*``)."""
    return haversine_km(lat, lon, config.CITY_CENTER_LAT, config.CITY_CENTER_LON)
