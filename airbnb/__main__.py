"""``python -m airbnb`` entry point — dispatches to the unified CLI."""

from __future__ import annotations

from airbnb.cli.app import main

if __name__ == "__main__":
    raise SystemExit(main())
