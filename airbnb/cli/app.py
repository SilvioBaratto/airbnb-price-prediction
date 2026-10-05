"""The unified ``airbnb`` command-line interface (composition root).

Four subcommands:

* ``fetch-data`` — download the Inside Airbnb dump and rebuild the cleaned snapshot (forwards to
  :func:`airbnb.datasource.insideairbnb.main`).
* ``run-arc`` — run the eight-part tree-ensemble arc (forwards to :func:`airbnb.modeling.arc.main`).
* ``train`` — fit the price models once and save them to disk
  (:meth:`airbnb.infrastructure.predictors.ModelPredictor.save`).
* ``simulate`` — interactive nightly-price simulator; wires the place repository, a
  :class:`~airbnb.infrastructure.predictors.ModelPredictor` (loaded from ``--model`` when given,
  else fitted on launch) and the :class:`~airbnb.application.pricing.PricingService`, then runs
  the terminal loop.

This module is the only place the layers are wired together; everything else depends inward.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from airbnb import paths


def _handle_fetch_data(args: argparse.Namespace) -> int:
    from airbnb.datasource import insideairbnb

    insideairbnb.main(args.forward)
    return 0


def _handle_run_arc(args: argparse.Namespace) -> int:
    from airbnb.modeling import arc

    arc.main(args.forward)
    return 0


def run_train(
    listings_path: Path | str,
    out_path: Path | str,
    *,
    out: Callable[[str], None] = print,
) -> int:
    """Fit the price models on ``listings_path`` and save them to ``out_path``.

    The seam behind ``train``: ``out`` is injected so tests can capture the messages.
    """
    from airbnb.infrastructure.predictors import ModelPredictor

    out(f"Training the price models on {listings_path} ...")
    predictor = ModelPredictor(listings_path)
    saved = predictor.save(out_path)
    out(f"Models trained on {predictor.n_train:,} listings -> saved to {saved}")
    return 0


def _handle_train(args: argparse.Namespace) -> int:
    return run_train(args.listings, args.out)


def run_simulate(
    listings_path: Path | str,
    *,
    model_path: Path | str | None = None,
    places_path: Path | str | None = None,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
) -> int:
    """Wire the simulate composition and run the loop (the testable seam behind ``simulate``).

    If ``model_path`` exists the models are loaded from it; otherwise they are fitted on launch
    from ``listings_path`` (a few seconds on the committed snapshot). ``input_fn``/``out`` are
    injected so an end-to-end run can be driven and captured by tests.
    """
    from airbnb.application.pricing import PricingService
    from airbnb.cli.simulator import run_simulator
    from airbnb.infrastructure.predictors import ModelPredictor
    from airbnb.infrastructure.repositories import CsvListingRepository, CsvPlaceRepository
    from airbnb.modeling.pipeline import load_listings

    snapshot = load_listings(listings_path)
    places = CsvPlaceRepository(places_path, listings=snapshot)
    listings = CsvListingRepository(listings_path)
    if model_path is not None and Path(model_path).exists():
        out(f"Loading saved price models from {model_path} ...")
        predictor = ModelPredictor.load(model_path)
        out(f"Models loaded (trained on {predictor.n_train:,} listings).\n")
    else:
        if model_path is not None:
            out(f"No saved models at {model_path}; training on launch instead.")
        out(f"Training the price models on {listings_path} (train-on-launch) ...")
        predictor = ModelPredictor(listings=snapshot)
        out(f"Models trained on {predictor.n_train:,} listings.\n")
    run_simulator(PricingService(predictor, listings), places, input_fn=input_fn, out=out)
    return 0


def _handle_simulate(args: argparse.Namespace) -> int:
    return run_simulate(args.listings, model_path=args.model, places_path=args.places)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``airbnb`` argument parser with its four subcommands."""
    parser = argparse.ArgumentParser(
        prog="airbnb",
        description="Airbnb nightly-price toolkit: fetch the data, run the modeling arc, "
        "train the price models, or price a listing.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser(
        "fetch-data",
        add_help=False,
        help="download Inside Airbnb and rebuild the snapshot (see: airbnb fetch-data --help)",
    )
    fetch.set_defaults(func=_handle_fetch_data)

    arc_cmd = sub.add_parser(
        "run-arc",
        add_help=False,
        help="run the eight-part tree-ensemble arc (see: airbnb run-arc --help)",
    )
    arc_cmd.set_defaults(func=_handle_run_arc)

    listings_help = "cleaned snapshot to fit on (default: data/raw/listings_rome.csv)"
    train_cmd = sub.add_parser("train", help="fit the price models and save them to disk")
    train_cmd.add_argument("--listings", type=Path, default=paths.LISTINGS_CSV, help=listings_help)
    train_cmd.add_argument(
        "--out",
        type=Path,
        default=paths.MODEL_JOBLIB,
        help=f"where to save the models (default: {paths.MODEL_JOBLIB})",
    )
    train_cmd.set_defaults(func=_handle_train)

    sim = sub.add_parser("simulate", help="interactive nightly-price simulator")
    sim.add_argument("--listings", type=Path, default=paths.LISTINGS_CSV, help=listings_help)
    sim.add_argument(
        "--model",
        type=Path,
        default=None,
        help="load saved models from this path instead of training on launch "
        f"(e.g. {paths.MODEL_JOBLIB}); falls back to training if the file is missing",
    )
    sim.add_argument(
        "--places",
        type=Path,
        default=paths.PLACES_CSV,
        help="the places a host can pick (default: data/sources/rome_places.csv)",
    )
    sim.set_defaults(func=_handle_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand's handler.

    ``fetch-data`` and ``run-arc`` forward their options to their module CLIs, so unknown args
    are collected (``args.forward``) rather than rejected; ``train`` and ``simulate`` are strict.
    """
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.command in {"simulate", "train"} and extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.forward = extra
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
