"""The terminal layer: rendering, the interactive loop, the parser, and the end-to-end wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from airbnb.application.pricing import PricingService
from airbnb.cli import app
from airbnb.cli.render import describe, render_comparables, render_quote
from airbnb.cli.simulator import run_simulator
from airbnb.domain.entities import Comparable, ListingRequest, Place, PriceQuote
from airbnb.infrastructure.predictors import MedianPredictor
from airbnb.infrastructure.repositories import CsvListingRepository, CsvPlaceRepository

PLACE = Place(1, "Trastevere", "I Centro Storico", 41.8894, 12.4700)


def _drive(
    answers: list[str],
    pricing: PricingService,
    places: CsvPlaceRepository,
    *,
    then: type[BaseException] = EOFError,
) -> str:
    """Run the simulator on scripted answers and return everything it printed.

    Once the answers run out, the next prompt raises ``then``: ``EOFError`` is what ``input()``
    does when piped stdin ends or the user presses Ctrl-D, ``KeyboardInterrupt`` is Ctrl-C.
    """
    feed = iter(answers)

    def answer(_prompt: str) -> str:
        try:
            return next(feed)
        except StopIteration:
            raise then from None

    lines: list[str] = []
    run_simulator(pricing, places, input_fn=answer, out=lines.append)
    return "\n".join(lines)


@pytest.fixture
def wiring(places_csv: Path, listings: pd.DataFrame) -> tuple[PricingService, CsvPlaceRepository]:
    """A pricing service on the median fallback, with real repositories over synthetic data."""
    places = CsvPlaceRepository(places_csv, listings=listings)
    pricing = PricingService(MedianPredictor(listings), CsvListingRepository(listings=listings))
    return pricing, places


# --- render -------------------------------------------------------------------
def test_describe_pluralizes_counts() -> None:
    """ "1 bed" but "2 guests" and "1.5 bathrooms"."""
    text = describe(ListingRequest(PLACE, "Entire home/apt", 2, 1, 1, 1.5))
    assert text == "Trastevere · Entire home/apt · 2 guests · 1 bedroom · 1 bed · 1.5 bathrooms"


def test_render_quote_states_price_and_band() -> None:
    """The sentence carries the rounded price and both ends of the band."""
    text = render_quote(PriceQuote(152.4, "Private room", 99.6, 1210.0))
    assert "€152 a night" in text and "€100 – €1,210" in text


def test_render_comparables_aligns_rows_and_marks_unknown_counts() -> None:
    """Unknown bedrooms print as "?", and every row has the header's width or less."""
    table = render_comparables([Comparable(42, 0.05, "Private room", 2, None, 1.0, 88.0)])
    header, rule, row = table.splitlines()
    assert header.startswith("Distance") and set(rule.replace(" ", "")) == {"-"}
    assert "?" in row and "€88" in row and row.endswith("42")


def test_render_comparables_without_rows_still_prints_the_header() -> None:
    """An empty result is never a blank line."""
    assert render_comparables([]).startswith("Distance")


# --- simulator ------------------------------------------------------------------
def test_simulator_prices_a_listing_and_shows_comparables(wiring) -> None:
    """Name search, room type by number, defaults on Enter, then quit."""
    text = _drive(["citta", "2", "", "", "", "", "q"], *wiring)
    assert "-> Città Vecchia (I Centro Storico" in text
    assert "Città Vecchia · Private room · 2 guests" in text
    assert "Suggested price: €" in text
    assert "The nearest real listings like it:" in text
    assert text.endswith("Ciao!")


def test_simulator_lists_places_and_recovers_from_bad_input(wiring) -> None:
    """'?' lists places; unknown places, room types and numbers re-prompt; quit works anywhere."""
    text = _drive(["?", "atlantis", "3", "castle", "entire", "two", "0", "", "", "", "q"], *wiring)
    assert " 2  Parioli" in text
    assert "No place matches 'atlantis'" in text
    assert "'castle' is not a room type" in text
    assert "'two' is not a number" in text
    assert "guests must be between 1 and 16" in text
    assert text.endswith("Ciao!")


def test_simulator_quits_from_a_number_prompt(wiring) -> None:
    """Typing q mid-listing ends the session without a traceback."""
    text = _drive(["1", "", "quit"], *wiring)
    assert "Suggested price" not in text and text.endswith("Ciao!")


@pytest.mark.parametrize("ending", [EOFError, KeyboardInterrupt])
def test_simulator_leaves_cleanly_on_end_of_input_or_ctrl_c(wiring, ending) -> None:
    """Ctrl-D, an exhausted pipe or Ctrl-C mid-listing end the session with no traceback."""
    text = _drive(["citta", "2"], *wiring, then=ending)
    assert "Suggested price" not in text
    assert text.endswith("\nCiao!")
    assert text.splitlines()[-2] == ""


def test_simulator_insists_on_whole_finite_counts(wiring) -> None:
    """Half a guest, NaN and infinity are refused instead of truncated or crashing."""
    text = _drive(["citta", "", "2.5", "nan", "inf", "3", "", "", "1,5", "q"], *wiring)
    assert "'2.5' is not a whole number" in text
    assert "'nan' is not a number" in text and "'inf' is not a number" in text
    assert "3 guests" in text and "1.5 bathrooms" in text


def test_simulator_does_not_crash_on_digits_int_cannot_read(wiring) -> None:
    """A superscript two, or 5,000 digits (past int()'s limit), is a failed lookup, not a crash."""
    huge = "9" * 5000
    text = _drive(["²", huge, "citta", huge, "q"], *wiring)
    assert "No place matches '²'" in text
    assert text.count("No place matches") == 2
    assert "is not a room type" in text and text.endswith("Ciao!")


# --- parser -------------------------------------------------------------------
def test_parser_knows_the_four_subcommands() -> None:
    """Every documented subcommand parses."""
    parser = app.build_parser()
    for command in ("fetch-data", "run-arc", "train", "simulate"):
        args, _ = parser.parse_known_args([command])
        assert args.command == command


def test_train_and_simulate_reject_unknown_flags() -> None:
    """The strict subcommands fail loudly on typos instead of ignoring them."""
    with pytest.raises(SystemExit):
        app.main(["train", "--bogus"])


@pytest.mark.parametrize(
    "command, module, attr",
    [
        ("fetch-data", "airbnb.datasource.insideairbnb", "main"),
        ("run-arc", "airbnb.modeling.arc", "main"),
    ],
)
def test_forwarding_subcommands_pass_their_flags_through(
    command: str, module: str, attr: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fetch-data`` and ``run-arc`` hand their options to the module CLIs untouched."""
    seen: list[list[str]] = []
    monkeypatch.setattr(f"{module}.{attr}", lambda argv: seen.append(argv))
    assert app.main([command, "--seed", "3"]) == 0
    assert seen == [["--seed", "3"]]


# --- end to end -----------------------------------------------------------------
def test_train_then_simulate_loads_the_saved_models(
    snapshot_csv: Path, places_csv: Path, tmp_path: Path
) -> None:
    """``train`` writes the weights; ``simulate --model`` loads them without refitting."""
    model_path = tmp_path / "models" / "price.joblib"
    lines: list[str] = []
    assert app.run_train(snapshot_csv, model_path, out=lines.append) == 0
    assert model_path.exists() and "saved to" in lines[-1]

    feed = iter(["parioli", "", "4", "", "", "", "q"])
    out: list[str] = []
    app.run_simulate(
        snapshot_csv,
        model_path=model_path,
        places_path=places_csv,
        input_fn=lambda _prompt: next(feed),
        out=out.append,
    )
    text = "\n".join(out)
    assert "Models loaded" in text and "train-on-launch" not in text
    assert "Parioli · Entire home/apt · 4 guests" in text
    assert "Suggested price: €" in text


def test_simulate_without_saved_models_trains_on_launch(
    snapshot_csv: Path, places_csv: Path, tmp_path: Path
) -> None:
    """A missing --model path falls back to fitting on the snapshot instead of crashing."""
    feed = iter(["q"])
    out: list[str] = []
    app.run_simulate(
        snapshot_csv,
        model_path=tmp_path / "missing.joblib",
        places_path=places_csv,
        input_fn=lambda _prompt: next(feed),
        out=out.append,
    )
    text = "\n".join(out)
    assert "No saved models" in text and "Models trained on 400 listings" in text
