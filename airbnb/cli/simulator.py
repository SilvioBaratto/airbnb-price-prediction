"""Interactive terminal simulator — describe a listing in Rome, see its nightly price.

The loop asks for a place (by name or number), the room type, then guests, bedrooms, beds and
bathrooms; it prints the suggested price from the injected
:class:`~airbnb.application.pricing.PricingService` and the real listings nearby that look like
it, until the user quits with ``q``, Ctrl-D or Ctrl-C. I/O is injected (``input_fn`` / ``out``)
so the loop is driven and captured in tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from airbnb.application.ports import PlaceRepository
from airbnb.application.pricing import PricingService
from airbnb.cli.render import describe, render_comparables, render_quote
from airbnb.domain import config, geo
from airbnb.domain.entities import ListingRequest, Place

_QUIT = {"q", "quit", "exit"}
_LIST = {"?", "list", "ls"}
_INTRO = (
    f"Airbnb {config.CITY} — nightly price simulator. "
    "Pick a place by name or number ('?' lists them, 'q' quits)."
)


class _Quit(Exception):
    """Raised by a prompt when the user leaves: a quit word, end of input, or Ctrl-C.

    Attributes:
        mid_line: the input ended without a newline, so the cursor still sits on the prompt.
    """

    def __init__(self, mid_line: bool = False) -> None:
        super().__init__()
        self.mid_line = mid_line


def _ask(prompt: str, input_fn: Callable[[str], str]) -> str:
    try:
        raw = input_fn(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise _Quit(mid_line=True) from None
    if raw.lower() in _QUIT:
        raise _Quit
    return raw


def _as_int(raw: str) -> int | None:
    """Read a typed whole number, or ``None``.

    ``isdecimal`` rather than ``isdigit``, since "²" is a digit ``int`` rejects, and the
    ``ValueError`` guard because ``int`` also refuses strings past 4,300 digits.
    """
    if not raw.isdecimal():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _resolve(places: PlaceRepository, raw: str) -> Place | None:
    """Resolve typed text to a place: a number is an id, anything else a name search."""
    number = _as_int(raw)
    if number is not None:
        try:
            return places.get(number)
        except KeyError:
            return None
    hits = places.search(raw)
    return hits[0] if hits else None


def _prompt_place(
    places: PlaceRepository, input_fn: Callable[[str], str], out: Callable[[str], None]
) -> Place:
    while True:
        raw = _ask("Place: ", input_fn)
        if raw.lower() in _LIST or not raw:
            for p in places.all():
                out(f"  {p.place_id:>2}  {p.name}")
            continue
        place = _resolve(places, raw)
        if place is None:
            out(f"  No place matches {raw!r}. Type '?' to list them.")
            continue
        km = geo.km_to_center(place.lat, place.lon)
        where = f"{place.neighbourhood}, {km:.1f} km from {config.CITY_CENTER_NAME}"
        out(f"  -> {place.name} ({where})")
        return place


def _prompt_room_type(input_fn: Callable[[str], str], out: Callable[[str], None]) -> str:
    choices = list(config.ROOM_TYPES)
    menu = ", ".join(f"{i} {name}" for i, name in enumerate(choices, start=1))
    while True:
        raw = _ask(f"Room type ({menu}) [1]: ", input_fn)
        if not raw:
            return choices[0]
        number = _as_int(raw)
        if number is not None and 1 <= number <= len(choices):
            return choices[number - 1]
        hits = [c for c in choices if c.lower().startswith(raw.lower())]
        if hits:
            return hits[0]
        out(f"  {raw!r} is not a room type.")


def _parse_number(raw: str) -> float | None:
    """Read a typed number, decimal comma allowed; ``None`` for text, NaN and infinity."""
    try:
        value = float(raw.replace(",", "."))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _prompt_number(
    label: str,
    default: float,
    input_fn: Callable[[str], str],
    out: Callable[[str], None],
    *,
    whole: bool,
) -> float:
    while True:
        raw = _ask(f"{label} [{default:g}]: ", input_fn)
        if not raw:
            return default
        value = _parse_number(raw)
        if value is None:
            out(f"  {raw!r} is not a number.")
        elif whole and not value.is_integer():
            out(f"  {raw!r} is not a whole number.")
        else:
            return value


def _prompt_request(
    places: PlaceRepository, input_fn: Callable[[str], str], out: Callable[[str], None]
) -> ListingRequest:
    place = _prompt_place(places, input_fn, out)
    room_type = _prompt_room_type(input_fn, out)
    while True:
        guests = int(_prompt_number("Guests", 2, input_fn, out, whole=True))
        bedrooms = int(_prompt_number("Bedrooms", max(1, guests // 2), input_fn, out, whole=True))
        beds = int(_prompt_number("Beds", max(1, (guests + 1) // 2), input_fn, out, whole=True))
        bathrooms = _prompt_number("Bathrooms", 1, input_fn, out, whole=False)
        try:
            return ListingRequest(place, room_type, guests, bedrooms, beds, bathrooms)
        except ValueError as exc:
            out(f"  {exc}. Let's try again.")


def run_simulator(
    pricing: PricingService,
    places: PlaceRepository,
    *,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> None:
    """Run the interactive pricing loop until the user quits.

    Args:
        pricing: prices each listing the user describes.
        places: resolves the place the user types.
        input_fn: reads one line of user input given a prompt.
        out: writes one line of output.
    """
    out(_INTRO)
    while True:
        try:
            request = _prompt_request(places, input_fn, out)
        except _Quit as quit_:
            if quit_.mid_line:
                out("")
            break
        out(f"\n{describe(request)}")
        out(render_quote(pricing.quote(request)))
        comparables = pricing.comparables(request)
        if comparables:
            out("\nThe nearest real listings like it:")
            out(render_comparables(comparables))
        out("")
    out("Ciao!")
