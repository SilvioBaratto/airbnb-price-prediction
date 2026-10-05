"""Render a quote and its comparables for the terminal (presentation layer).

Pure string formatting — no I/O. The caller (:mod:`airbnb.cli.simulator`) decides where to
print it.
"""

from __future__ import annotations

from airbnb.domain.entities import Comparable, ListingRequest, PriceQuote

_GUTTER = "  "


def _plural(n: float, word: str) -> str:
    shown = f"{n:g}"
    return f"{shown} {word}" if shown == "1" else f"{shown} {word}s"


def describe(request: ListingRequest) -> str:
    """Return a one-line summary (``Trastevere · Entire home/apt · 2 guests · ...``)."""
    return " · ".join(
        [
            request.place.name,
            request.room_type,
            _plural(request.guests, "guest"),
            _plural(request.bedrooms, "bedroom"),
            _plural(request.beds, "bed"),
            _plural(request.bathrooms, "bathroom"),
        ]
    )


def render_quote(quote: PriceQuote) -> str:
    """Return the suggestion as one sentence, band included."""
    return (
        f"Suggested price: €{quote.price_eur:,.0f} a night "
        f"(80% of similar listings ask €{quote.low_eur:,.0f} – €{quote.high_eur:,.0f})"
    )


def _table(headers: tuple[str, ...], align: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def _line(cells: tuple[str, ...]) -> str:
        cells_out = (format(cell, f"{align[i]}{widths[i]}") for i, cell in enumerate(cells))
        return _GUTTER.join(cells_out).rstrip()

    rule = tuple("-" * w for w in widths)
    return "\n".join([_line(headers), _line(rule), *(_line(r) for r in rows)])


def _count(value: float | None) -> str:
    return "?" if value is None else f"{value:g}"


def render_comparables(comparables: list[Comparable]) -> str:
    """Format real nearby listings as an aligned table, nearest first.

    An empty list still renders the header, so the output is never blank.
    """
    rows = [
        (
            f"{c.km:.2f} km",
            str(c.guests),
            _count(c.bedrooms),
            _count(c.bathrooms),
            f"€{c.price_eur:,.0f}",
            str(c.listing_id),
        )
        for c in comparables
    ]
    headers = ("Distance", "Guests", "Bedrooms", "Baths", "Price", "Listing id")
    return _table(headers, (">", ">", ">", ">", ">", "<"), rows)
