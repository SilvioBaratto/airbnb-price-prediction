"""Pure domain value objects (no I/O).

``Place`` is a spot in Rome a host can pick; ``ListingRequest`` is the listing a host describes;
``PriceQuote`` is the suggested nightly price for it; ``Comparable`` is a real listing nearby
that looks like it. Validation lives in ``__post_init__`` so an invalid request can never reach
the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from airbnb.domain import config

MAX_GUESTS = 16  # Airbnb's own cap on "accommodates"


@dataclass(frozen=True, slots=True)
class Place:
    """A named spot in the city and the administrative area (municipio) it falls in."""

    place_id: int
    name: str
    neighbourhood: str
    lat: float
    lon: float


@dataclass(frozen=True, slots=True)
class ListingRequest:
    """The listing a host wants priced: where it is, what kind of space, how big.

    Raises:
        ValueError: if the room type is not one of Airbnb's, or a count is out of the range
            Airbnb itself accepts.
    """

    place: Place
    room_type: str
    guests: int
    bedrooms: int
    beds: int
    bathrooms: float

    def __post_init__(self) -> None:
        if self.room_type not in config.ROOM_TYPES:
            raise ValueError(f"room type must be one of {config.ROOM_TYPES}, got {self.room_type}")
        if not 1 <= self.guests <= MAX_GUESTS:
            raise ValueError(f"guests must be between 1 and {MAX_GUESTS}, got {self.guests}")
        # A studio has zero bedrooms, so zero is a legitimate answer here.
        if not 0 <= self.bedrooms <= MAX_GUESTS:
            raise ValueError(f"bedrooms must be between 0 and {MAX_GUESTS}, got {self.bedrooms}")
        if not 1 <= self.beds <= MAX_GUESTS:
            raise ValueError(f"beds must be between 1 and {MAX_GUESTS}, got {self.beds}")
        if not 0 <= self.bathrooms <= MAX_GUESTS or (self.bathrooms * 2) % 1:
            raise ValueError(f"bathrooms must be a multiple of 0.5, got {self.bathrooms}")


@dataclass(frozen=True, slots=True, order=True)
class PriceQuote:
    """A suggested nightly price, with the band 80% of similar listings fall in.

    Ordering compares ``price_eur`` only, so ``sorted(quotes)`` lists the cheapest first.

    Raises:
        ValueError: if the price is not positive or falls outside its own band.
    """

    price_eur: float
    room_type: str = field(compare=False)
    low_eur: float = field(compare=False)
    high_eur: float = field(compare=False)

    def __post_init__(self) -> None:
        if self.price_eur <= 0:
            raise ValueError(f"price must be positive, got {self.price_eur}")
        if not self.low_eur <= self.price_eur <= self.high_eur:
            raise ValueError(
                f"band [{self.low_eur}, {self.high_eur}] does not contain {self.price_eur}"
            )


@dataclass(frozen=True, slots=True)
class Comparable:
    """A real listing near the requested place, with the price it actually asks."""

    listing_id: int
    km: float
    room_type: str
    guests: int
    bedrooms: float | None
    bathrooms: float | None
    price_eur: float
