"""Export compact JSON fixtures for the neuroespresso «Costruiamo l'algoritmo di Airbnb» videos.

Every number a video shows traces back to a real run. The curves and the summary table come from
what ``airbnb run-arc`` wrote to ``output/``, checked against the seed and snapshot recorded in
``arc_run.json``. The per-scene details (the depth-3 tree, fifty trees' verdicts on one listing,
which trees never saw a listing, the residuals shrinking round by round) are recomputed here with
the arc's own pipelines and seed, so they come from the same models the arc measured. Nothing is
hand-typed.

One JSON per scene family, keys in Italian camelCase like the Uber fixtures, each with a
``fonte`` block saying where its numbers come from.

Usage::

    airbnb run-arc                                 # once: output/*.csv + arc_run.json
    python scripts/export_fixtures.py              # -> output/fixtures/airbnb_prezzi/
    python scripts/export_fixtures.py --out-dir <videocraft>/public/fixtures/airbnb_prezzi
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Runnable as a plain script from a checkout, without `pip install -e .` first.
sys.path.insert(0, str(PROJECT_ROOT))

from airbnb.application.pricing import PricingService
from airbnb.domain import config, geo
from airbnb.domain.entities import ListingRequest
from airbnb.infrastructure import paths
from airbnb.infrastructure.predictors import ModelPredictor
from airbnb.infrastructure.repositories import (
    CsvListingRepository,
    CsvPlaceRepository,
)
from airbnb.modeling import arc, pipeline

SLUG = "airbnb_prezzi"
DATI = "Inside Airbnb, Roma, 2026-06-20 (CC BY 4.0)"

MAP_N = 2000  # listing dots on the Rome map: enough to show the centre's density, cheap to ship
TREE_ROWS_N = 120  # listings that file through the depth-3 tree on screen
VERDICTS_N = 50  # trees whose verdicts on one listing the video lines up
BOOTSTRAP_ROWS = 10  # a table small enough to read while it is redrawn with replacement
BOOTSTRAP_DRAWS = 3
RESIDUAL_ROUNDS = (1, 2, 3, 5, 10, 25, 50, 100)
CURVE_POINTS = 80  # the boosting curve has 3,000 rounds; log-spaced samples keep its shape
SHUFFLE_ROWS = 8
# The README's example, so the project video and the repo show the same quote.
EXAMPLE = {
    "luogo": "Trastevere",
    "room_type": "Entire home/apt",
    "guests": 2,
    "bedrooms": 1,
    "beds": 1,
    "bathrooms": 1.0,
}


class ArcOutputMissing(Exception):
    """An ``airbnb run-arc`` artefact is missing, or comes from another seed or snapshot."""


def _num(x: Any, nd: int = 2) -> float | None:
    """Round ``x`` for shipping, mapping NaN (pandas' blank cell) to JSON null."""
    if x is None or pd.isna(x):
        return None
    return round(float(x), nd)


def _cell(x: Any) -> float | str | None:
    """Ship a table cell as it reads: a category as text, a number rounded."""
    return x if isinstance(x, str) else _num(x)


def _listing(row: pd.Series) -> dict[str, Any]:
    """The few facts about a listing a scene can print next to it."""
    return {
        "tipo": str(row["room_type"]),
        "municipio": str(row["neighbourhood"]),
        "ospiti": int(row["accommodates"]),
        "camere": _num(row["bedrooms"], 0),
        "bagni": _num(row["bathrooms"], 1),
        "kmDalCentro": _num(row["km_to_center"]),
    }


@dataclass
class Contesto:
    """The snapshot, the arc's split, and lazily fitted models shared across fixtures."""

    listings: pd.DataFrame
    split: pipeline.Split
    seed: int
    arc_dir: Path
    listings_path: Path
    places_path: Path

    def fonte(self, **extra: Any) -> dict[str, Any]:
        """Return the provenance block every fixture starts with."""
        return {
            "dati": DATI,
            "seme": self.seed,
            "nAnnunci": len(self.listings),
            "nTrain": len(self.split.y_train),
            "nTest": len(self.split.y_test),
            **extra,
        }

    def rng(self) -> np.random.Generator:
        """Return a fresh generator, so each fixture is reproducible on its own."""
        return np.random.default_rng(self.seed)

    @cached_property
    def arc_run(self) -> dict[str, Any]:
        """Return ``arc_run.json``, refusing a run made with another seed or snapshot.

        Raises:
            ArcOutputMissing: if the file is absent or describes a different run.
        """
        path = self.arc_dir / "arc_run.json"
        if not path.exists():
            raise ArcOutputMissing(f"{path} missing")
        run = json.loads(path.read_text())
        if run["seed"] != self.seed or run["n_listings"] != len(self.listings):
            raise ArcOutputMissing(
                f"{path} is from seed {run['seed']} on {run['n_listings']:,} listings, "
                f"not seed {self.seed} on {len(self.listings):,}"
            )
        return run

    def arc_csv(self, name: str) -> pd.DataFrame:
        """Read one of the arc's CSVs, after checking the run it belongs to.

        Raises:
            ArcOutputMissing: if the run is stale or the CSV is absent.
        """
        self.arc_run  # noqa: B018  (validates the run before trusting its CSVs)
        path = self.arc_dir / name
        if not path.exists():
            raise ArcOutputMissing(f"{path} missing")
        return pd.read_csv(path)

    def arc_mae(self, part: int, contains: str = "") -> float:
        """Return the test MAE the arc reported for ``part``, on the row naming ``contains``."""
        summary = self.arc_csv("arc_summary.csv")
        row = summary[(summary["part"] == part) & summary["name"].str.contains(contains)]
        return float(row["mae_test"].iloc[0])

    @cached_property
    def train_frame(self) -> pd.DataFrame:
        """The training rows with their prices, shaped like the snapshot."""
        return self.split.X_train.assign(**{config.TARGET: self.split.y_train})

    @cached_property
    def bagged(self):
        """Part 3's bagging ensemble, refitted with the arc's size and seed."""
        model = pipeline.bagging_pipeline(seed=self.seed)
        return model.fit(self.split.X_train, self.split.y_train)

    @cached_property
    def forest(self):
        """Part 4's forest (OOB-scored), refitted with the arc's size and seed."""
        model = pipeline.forest_pipeline(oob_score=True, seed=self.seed)
        return model.fit(self.split.X_train, self.split.y_train)


# --- The project ---------------------------------------------------------------
def fx_repo(ctx: Contesto) -> dict[str, Any] | None:
    """The repository's own layout, as git tracks it, for the "how it is built" scene.

    The listing comes from ``git ls-files`` rather than from walking the disk: the working copy
    also holds ``.venv``, caches and the git-ignored raw dump, none of which is the project.
    """

    def _git(*argv: str) -> str:
        return subprocess.run(
            ["git", *argv], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        tracked = [Path(p) for p in _git("ls-files").splitlines()]
        commit = _git("rev-parse", "--short", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return None
    top = sorted(
        {(p.parts[0], "dir" if len(p.parts) > 1 else "file") for p in tracked},
        key=lambda x: (x[1] == "file", x[0]),
    )
    layers = sorted({p.parts[1] for p in tracked if p.parts[0] == "airbnb" and len(p.parts) > 2})
    strati = []
    for name in layers:
        files = sorted(p.name for p in tracked if p.parts[:2] == ("airbnb", name))
        strati.append({"nome": name, "file": files, "nFile": len(files)})
    return {
        "repo": PROJECT_ROOT.name,
        "commit": commit,
        "pacchetto": "airbnb",
        "nFileTracciati": len(tracked),
        "nCartellePacchetto": len(strati),
        "nFilePacchetto": sum(1 for p in tracked if p.parts[0] == "airbnb" and len(p.parts) > 1),
        "primoLivello": [{"nome": n, "tipo": t} for n, t in top],
        "strati": strati,
    }


def fx_arc_metrics(ctx: Contesto) -> dict[str, Any]:
    """The error spine: the arc's summary table, plus the simulator's own model.

    The simulator's model is not a row of the arc, so it is fitted here on the arc's training
    rows and scored on its test rows, instead of quoting a number from elsewhere.
    """
    summary = ctx.arc_csv("arc_summary.csv")
    parti = [
        {
            "parte": int(r.part),
            "modello": str(r.name),
            "maeTrain": _num(r.mae_train),
            "maeTest": _num(r.mae_test),
            "r2Test": _num(r.r2_test, 3),
            "nota": "" if pd.isna(r.note) else str(r.note),
        }
        for r in summary.itertuples(index=False)
    ]
    predictor = ModelPredictor.from_frame(ctx.train_frame)
    price = predictor.predict(ctx.split.X_test)
    low, high = predictor.predict_band(ctx.split.X_test)
    y = ctx.split.y_test
    return {
        "fonte": ctx.fonte(tabella="output/arc_summary.csv"),
        "parti": parti,
        "simulatore": {
            "modello": "boosting, perdita assoluta, con banda 10-90% da due boosting quantili",
            "maeTest": _num(pipeline.mae(y, price)),
            "coperturaBanda": _num(np.mean((y >= low) & (y <= high)), 3),
            "larghezzaMedianaBanda": _num(np.median(high - low), 0),
        },
    }


def fx_mappa(ctx: Contesto) -> dict[str, Any]:
    """Listing dots on the Rome map, the simulator's places, and each municipio's median."""
    listings = ctx.listings
    idx = np.sort(ctx.rng().choice(len(listings), size=min(MAP_N, len(listings)), replace=False))
    sample = listings.iloc[idx]
    tipi = list(config.ROOM_TYPES)
    places = CsvPlaceRepository(ctx.places_path, listings=listings).all()
    hoods = (
        listings.groupby("neighbourhood")[config.TARGET]
        .agg(["count", "median"])
        .sort_values("median", ascending=False)
    )
    return {
        "fonte": ctx.fonte(campione=len(sample)),
        "centro": {
            "nome": config.CITY_CENTER_NAME,
            "lat": config.CITY_CENTER_LAT,
            "lon": config.CITY_CENTER_LON,
        },
        "tipi": tipi,
        "punti": {
            "lat": [round(float(v), 5) for v in sample["latitude"]],
            "lon": [round(float(v), 5) for v in sample["longitude"]],
            "prezzo": [round(float(v)) for v in sample[config.TARGET]],
            "tipo": [tipi.index(t) for t in sample["room_type"]],
        },
        "luoghi": [
            {
                "nome": p.name,
                "municipio": p.neighbourhood,
                "lat": p.lat,
                "lon": p.lon,
                "kmDalCentro": _num(geo.km_to_center(p.lat, p.lon)),
            }
            for p in places
        ],
        "municipi": [
            {"nome": str(name), "annunci": int(row["count"]), "prezzoMediano": _num(row["median"])}
            for name, row in hoods.iterrows()
        ],
    }


# --- Part 1 — the tree that cuts the table -------------------------------------------
def fx_albero(ctx: Contesto) -> dict[str, Any]:
    """The depth-3 tree node by node, the error at each depth, and listings walking through it.

    Row values are what the tree sees, so a missing bedroom count shows as the training median
    the pipeline imputed.
    """
    s = ctx.split
    livelli = [
        {
            "profondita": 0,
            "foglie": 1,
            "maeTest": _num(pipeline.mae(s.y_test, np.full_like(s.y_test, s.y_train.mean()))),
        }
    ]
    models = {
        depth: pipeline.tree_pipeline(max_depth=depth, seed=ctx.seed).fit(s.X_train, s.y_train)
        for depth in (1, 2, 3)
    }
    for depth, fitted in models.items():
        livelli.append(
            {
                "profondita": depth,
                "foglie": int(fitted.named_steps["est"].get_n_leaves()),
                "maeTest": _num(pipeline.score(fitted, s.X_test, s.y_test)[0]),
            }
        )
    model = models[3]
    est = model.named_steps["est"]
    tree = est.tree_
    names = pipeline.design_names(model)

    depth_of = {0: 0}
    for node in range(tree.node_count):
        for child in (tree.children_left[node], tree.children_right[node]):
            if child != -1:
                depth_of[int(child)] = depth_of[node] + 1
    nodi = []
    for node in range(tree.node_count):
        leaf = tree.children_left[node] == -1
        nodi.append(
            {
                "id": node,
                "profondita": depth_of[node],
                "colonna": None if leaf else names[tree.feature[node]],
                "soglia": None if leaf else _num(tree.threshold[node], 3),
                "annunci": int(tree.n_node_samples[node]),
                "prezzoMedio": _num(tree.value[node].ravel()[0]),
                "sinistra": None if leaf else int(tree.children_left[node]),
                "destra": None if leaf else int(tree.children_right[node]),
            }
        )

    used = sorted({n["colonna"] for n in nodi if n["colonna"] is not None})
    rows = np.sort(
        ctx.rng().choice(len(s.y_test), size=min(TREE_ROWS_N, len(s.y_test)), replace=False)
    )
    Xd = pipeline.design(model, s.X_test.iloc[rows])
    paths_ = est.decision_path(Xd)
    righe = [
        {
            "valori": {c: _num(Xd[i, names.index(c)]) for c in used},
            "prezzo": _num(s.y_test[r]),
            "percorso": [int(n) for n in paths_.indices[paths_.indptr[i] : paths_.indptr[i + 1]]],
            "stima": _num(tree.value[est.apply(Xd[i : i + 1])[0]].ravel()[0]),
        }
        for i, r in enumerate(rows)
    ]
    return {
        "fonte": ctx.fonte(criterio="somma dei quadrati", campioneRighe=len(righe)),
        "mediaTrain": _num(s.y_train.mean()),
        "livelli": livelli,
        "colonneUsate": used,
        "nodi": nodi,
        "righe": righe,
    }


# --- Part 2 — when to stop cutting --------------------------------------------------
def fx_potatura(ctx: Contesto) -> dict[str, Any]:
    """The memorizing full tree, and the pruning path with the leaf count CV picks."""
    s = ctx.split
    full = pipeline.tree_pipeline(seed=ctx.seed).fit(s.X_train, s.y_train)
    curve = ctx.arc_csv("part2_pruning.csv")
    best = curve.loc[curve["cv_mae"].idxmin()]

    def _point(r: Any) -> dict[str, Any]:
        return {
            "alpha": _num(r["alpha"], 4),
            "foglie": int(r["leaves"]),
            "maeCv": _num(r["cv_mae"]),
            "maeTrain": _num(r["train_mae"]),
            "maeTest": _num(r["test_mae"]),
        }

    return {
        "fonte": ctx.fonte(curva="output/part2_pruning.csv", foldCv=arc.CV_FOLDS),
        "alberoPieno": {
            "foglie": int(full.named_steps["est"].get_n_leaves()),
            "maeTrain": _num(pipeline.mae(s.y_train, full.predict(s.X_train))),
            "maeTest": _num(pipeline.score(full, s.X_test, s.y_test)[0]),
        },
        "curva": [_point(r) for _, r in curve.iterrows()],
        "scelto": _point(best),
    }


# --- Part 3 — redo the table by drawing lots ------------------------------------------
def fx_bagging(ctx: Contesto) -> dict[str, Any]:
    """A bootstrap redraw of a ten-row table, the error vs trees curve, and fifty verdicts.

    The verdicts are on the test listing whose bagged error is the median one: a typical
    listing, neither a showcase nor a failure.
    """
    s = ctx.split
    ensemble = ctx.bagged.named_steps["est"]
    n_train = len(s.y_train)
    share = float(np.mean([len(np.unique(b)) / n_train for b in ensemble.estimators_samples_]))
    per_tree = pipeline.tree_predictions(ctx.bagged, s.X_test)
    averaged = per_tree.mean(axis=0)
    typical = int(np.argsort(np.abs(averaged - s.y_test))[len(s.y_test) // 2])

    rng = ctx.rng()
    rows = np.sort(rng.choice(n_train, size=BOOTSTRAP_ROWS, replace=False))
    draws = [rng.integers(0, BOOTSTRAP_ROWS, BOOTSTRAP_ROWS) for _ in range(BOOTSTRAP_DRAWS)]
    curve = ctx.arc_csv("part3_bagging.csv")
    return {
        "fonte": ctx.fonte(alberi=len(ensemble.estimators_), curva="output/part3_bagging.csv"),
        "quotaRigheDistinte": _num(share, 4),
        "unoMenoUnoSuE": _num(1 - np.exp(-1), 4),
        "tabella": {
            "nota": "dimostrazione del ricampionamento su dieci annunci veri",
            "righe": [{**_listing(s.X_train.iloc[r]), "prezzo": _num(s.y_train[r])} for r in rows],
            "estrazioni": [
                {
                    "indici": d.tolist(),
                    "conteggi": np.bincount(d, minlength=BOOTSTRAP_ROWS).tolist(),
                }
                for d in draws
            ],
        },
        "alberoSolo": {"maeTest": _num(np.mean([pipeline.mae(s.y_test, p) for p in per_tree]))},
        "curva": [
            {"alberi": int(r.trees), "maeTest": _num(r.test_mae)}
            for r in curve.itertuples(index=False)
        ],
        "maeTest": _num(ctx.arc_mae(3)),
        "verdetti": {
            "annuncio": _listing(s.X_test.iloc[typical]),
            "prezzoVero": _num(s.y_test[typical]),
            "alberi": [round(float(p)) for p in per_tree[:VERDICTS_N, typical]],
            "mediaPrimi": _num(per_tree[:VERDICTS_N, typical].mean()),
            "mediaTutti": _num(averaged[typical]),
        },
    }


# --- Part 4 — trees that stop copying each other ----------------------------------------
def fx_foresta(ctx: Contesto) -> dict[str, Any]:
    """First cuts of bagged vs forest trees, their correlation, and the error vs ``m`` curve."""
    s = ctx.split
    n_cuts = arc.FIRST_SPLIT_SAMPLE
    bag_cuts = arc.first_cuts(ctx.bagged, n_cuts)
    forest_cuts = arc.first_cuts(ctx.forest, n_cuts)
    p = len(pipeline.column_owner(ctx.forest))
    b = len(ctx.forest.named_steps["est"].estimators_)

    def _correlation(make: Callable[[int, int], Any]) -> dict[str, Any]:
        rho, sigma2 = arc.tree_correlation(s, make, seed=ctx.seed)
        floor = rho * sigma2 + (1 - rho) * sigma2 / b
        return {
            "rho": _num(rho, 3),
            "sigma2": _num(sigma2, 0),
            "varianzaMedia": _num(floor, 0),
            "sdMedia": _num(floor**0.5, 1),
        }

    curve = ctx.arc_csv("part4_mtry.csv")
    return {
        "fonte": ctx.fonte(
            alberiConfrontati=n_cuts,
            insiemiDisgiunti=arc.CORRELATION_SETS,
            alberiPerInsieme=arc.CORRELATION_TREES,
            curva="output/part4_mtry.csv",
        ),
        "primoTaglio": {"bagging": bag_cuts, "foresta": forest_cuts},
        "correlazione": {
            "alberi": b,
            "bagging": _correlation(
                lambda n, sd: pipeline.bagging_pipeline(n_estimators=n, seed=sd)
            ),
            "foresta": _correlation(
                lambda n, sd: pipeline.forest_pipeline(n_estimators=n, seed=sd)
            ),
        },
        "colonnePerTaglio": {
            "p": p,
            "m": max(1, int(pipeline.FOREST_MAX_FEATURES * p)),
            "curva": [
                {"m": int(r.m), "maeTest": _num(r.test_mae)} for r in curve.itertuples(index=False)
            ],
        },
        "maeTest": {"bagging": _num(ctx.arc_mae(3)), "foresta": _num(ctx.arc_mae(4))},
    }


# --- Part 5 — the free test --------------------------------------------------------------
def fx_oob(ctx: Contesto) -> dict[str, Any]:
    """Listing #0 of the training rows: which trees drew it, and what the others say about it."""
    s = ctx.split
    ensemble = ctx.forest.named_steps["est"]
    draws = [int(np.sum(bag == 0)) for bag in ensemble.estimators_samples_]
    pred, n_trees = pipeline.out_of_bag_predict(ctx.forest, pipeline.design(ctx.forest, s.X_train))
    graded = n_trees > 0
    b = len(ensemble.estimators_)
    return {
        "fonte": ctx.fonte(alberi=b),
        "annuncio": _listing(s.X_train.iloc[0]),
        "prezzoVero": _num(s.y_train[0]),
        "estrazioniPerAlbero": draws[:VERDICTS_N],
        "alberiCheNonLoHannoVisto": int(sum(d == 0 for d in draws)),
        "stimaOob": _num(pred[0]),
        "quotaMediaNonVisto": _num(n_trees.mean() / b, 4),
        "unoSuE": _num(np.exp(-1), 4),
        "maeOob": _num(pipeline.mae(s.y_train[graded], pred[graded])),
        "maeTest": _num(ctx.arc_mae(4)),
    }


# --- Part 6 — which columns really matter ------------------------------------------------
def fx_importanza(ctx: Contesto) -> dict[str, Any]:
    """The permutation-importance ranking, and the top column shuffled on a few real rows."""
    s = ctx.split
    table = ctx.arc_csv("part6_importance.csv")
    pred, n_trees = pipeline.out_of_bag_predict(ctx.forest, pipeline.design(ctx.forest, s.X_train))
    graded = n_trees > 0
    top = str(table["column"].iloc[0])
    rng = ctx.rng()
    shuffled = s.X_train[top].to_numpy()[rng.permutation(len(s.y_train))]
    rows = np.sort(rng.choice(len(s.y_train), size=SHUFFLE_ROWS, replace=False))
    return {
        "fonte": ctx.fonte(tabella="output/part6_importance.csv"),
        "maeOob": _num(pipeline.mae(s.y_train[graded], pred[graded])),
        "colonne": [
            {
                "colonna": str(r.column),
                "aumentoMae": _num(r.mae_increase),
                "quotaGuadagno": _num(r.split_gain_share, 4),
            }
            for r in table.itertuples(index=False)
        ],
        "esempio": {
            "nota": "la colonna più importante, rimescolata fra gli annunci",
            "colonna": top,
            "righe": [
                {
                    "prezzo": _num(s.y_train[r]),
                    "prima": _cell(s.X_train[top].iloc[r]),
                    "dopo": _cell(shuffled[r]),
                }
                for r in rows
            ],
        },
    }


# --- Part 7 — the tree that learns from the previous one's mistakes ------------------------
def fx_boosting(ctx: Contesto) -> dict[str, Any]:
    """The error-vs-rounds curves, the chosen round counts, and residuals shrinking per listing.

    The residual scene uses the arc's squared-loss model (refitted with its chosen rounds) on
    five test listings spread across the price range. Round 0 is the mean, where boosting starts.
    """
    s = ctx.split
    curve = ctx.arc_csv("part7_boosting.csv")
    n = len(curve)
    picks = np.unique(np.round(np.geomspace(1, n, CURVE_POINTS)).astype(int)) - 1
    best = {label: int(curve[f"val_mae_{label}"].idxmin()) + 1 for label in ("squared", "absolute")}
    model = pipeline.boosting_pipeline(max_iter=best["squared"], seed=ctx.seed)
    model.fit(s.X_train, s.y_train)
    staged = list(model.named_steps["est"].staged_predict(pipeline.design(model, s.X_test)))
    order = np.argsort(s.y_test)
    chosen = [int(order[int(q * (len(order) - 1))]) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    rounds = [r for r in RESIDUAL_ROUNDS if r < best["squared"]] + [best["squared"]]

    def _migliore(label: str, name: str) -> dict[str, Any]:
        return {
            "alberi": best[label],
            "maeVal": _num(curve[f"val_mae_{label}"].iloc[best[label] - 1]),
            "maeTest": _num(ctx.arc_mae(7, name)),
        }

    return {
        "fonte": ctx.fonte(
            curva="output/part7_boosting.csv",
            foglie=pipeline.BOOSTING_LEAVES,
            tassoApprendimento=pipeline.BOOSTING_LEARNING_RATE,
        ),
        "curva": [
            {
                "alberi": int(i) + 1,
                "valQuadratica": _num(curve["val_mae_squared"].iloc[i]),
                "testQuadratica": _num(curve["test_mae_squared"].iloc[i]),
                "valAssoluta": _num(curve["val_mae_absolute"].iloc[i]),
                "testAssoluta": _num(curve["test_mae_absolute"].iloc[i]),
            }
            for i in picks
        ],
        "migliori": {
            "quadratica": _migliore("squared", "squared"),
            "assoluta": _migliore("absolute", "absolute"),
        },
        "foresta": {"maeTest": _num(ctx.arc_mae(4))},
        "residui": [
            {
                "annuncio": _listing(s.X_test.iloc[i]),
                "prezzoVero": _num(s.y_test[i]),
                "stime": [{"alberi": 0, "stima": _num(s.y_train.mean())}]
                + [{"alberi": r, "stima": _num(staged[r - 1][i])} for r in rounds],
            }
            for i in chosen
        ],
    }


# --- Part 8 — no method always wins -----------------------------------------------------
def fx_nfl(ctx: Contesto) -> dict[str, Any]:
    """The same eight methods ranked on Rome and on the diabetes study."""
    board = ctx.arc_csv("part8_nfl.csv")

    def _ranking(dataset: str) -> list[dict[str, Any]]:
        rows = board[board["dataset"] == dataset].sort_values("rank")
        return [
            {"posto": int(r.rank), "modello": str(r.model), "maeCv": _num(r.cv_mae)}
            for r in rows.itertuples(index=False)
        ]

    return {
        "fonte": ctx.fonte(tabella="output/part8_nfl.csv", foldCv=arc.CV_FOLDS),
        "roma": {"unita": "euro a notte", "classifica": _ranking("airbnb_rome")},
        "diabete": {
            "unita": "punti della scala di progressione",
            "descrizione": "442 pazienti, 10 variabili, progressione della malattia dopo un anno "
            "(Efron et al., 2004; incluso in scikit-learn)",
            "classifica": _ranking("diabetes"),
        },
    }


# --- The finished product ---------------------------------------------------------------
def fx_simulatore(ctx: Contesto) -> dict[str, Any]:
    """The README's example quote, from the same wiring ``airbnb simulate`` uses."""
    predictor = ModelPredictor.from_frame(ctx.listings)
    place = CsvPlaceRepository(ctx.places_path, listings=ctx.listings).search(EXAMPLE["luogo"])[0]
    service = PricingService(predictor, CsvListingRepository(ctx.listings_path))
    request = ListingRequest(
        place,
        EXAMPLE["room_type"],
        EXAMPLE["guests"],
        EXAMPLE["bedrooms"],
        EXAMPLE["beds"],
        EXAMPLE["bathrooms"],
    )
    quote = service.quote(request)
    return {
        "fonte": ctx.fonte(modello="ModelPredictor allenato su tutto lo snapshot"),
        "richiesta": {
            "luogo": place.name,
            "municipio": place.neighbourhood,
            "kmDalCentro": _num(geo.km_to_center(place.lat, place.lon)),
            "tipo": request.room_type,
            "ospiti": request.guests,
            "camere": request.bedrooms,
            "letti": request.beds,
            "bagni": request.bathrooms,
        },
        "prezzo": _num(quote.price_eur, 0),
        "bandaBassa": _num(quote.low_eur, 0),
        "bandaAlta": _num(quote.high_eur, 0),
        "simili": [
            {
                "id": c.listing_id,
                "km": _num(c.km, 3),
                "ospiti": c.guests,
                "camere": _num(c.bedrooms, 0),
                "bagni": _num(c.bathrooms, 1),
                "prezzo": _num(c.price_eur, 0),
            }
            for c in service.comparables(request)
        ],
    }


FIXTURES: dict[str, Callable[[Contesto], Any]] = {
    "repo.json": fx_repo,
    "arc_metrics.json": fx_arc_metrics,
    "mappa.json": fx_mappa,
    "albero.json": fx_albero,
    "potatura.json": fx_potatura,
    "bagging.json": fx_bagging,
    "foresta.json": fx_foresta,
    "oob.json": fx_oob,
    "importanza.json": fx_importanza,
    "boosting.json": fx_boosting,
    "nfl.json": fx_nfl,
    "simulatore.json": fx_simulatore,
}


def write_json(out_dir: Path, name: str, obj: object) -> Path:
    """Write ``obj`` as compact JSON to ``out_dir/name``, report its size, and return the path."""
    path = out_dir / name
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
    print(f"  {name:<20} {path.stat().st_size / 1024:6.1f} KB")
    return path


def main(argv: list[str] | None = None) -> dict[str, Path]:
    """Write every fixture whose sources are available; return the files written.

    Fixtures that need ``airbnb run-arc`` output are skipped, with the reason, when that output
    is missing or was produced with another seed or snapshot.
    """
    parser = argparse.ArgumentParser(
        description="Export JSON fixtures for the Airbnb-video animations."
    )
    parser.add_argument(
        "--listings", type=Path, default=paths.LISTINGS_CSV, help="cleaned snapshot"
    )
    parser.add_argument(
        "--arc-dir",
        type=Path,
        default=paths.OUTPUT_DIR,
        help="where `airbnb run-arc` wrote its CSVs and arc_run.json (default: output/)",
    )
    parser.add_argument(
        "--places", type=Path, default=paths.PLACES_CSV, help="the simulator's places"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.OUTPUT_DIR / "fixtures" / SLUG,
        help=f"destination (point at videocraft/public/fixtures/{SLUG} to write in place)",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED, help="the arc's seed (default: %(default)s)"
    )
    args = parser.parse_args(argv)

    listings = pipeline.load_listings(args.listings)
    ctx = Contesto(
        listings=listings,
        split=pipeline.make_split(listings, seed=args.seed),
        seed=args.seed,
        arc_dir=args.arc_dir,
        listings_path=args.listings,
        places_path=args.places,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing fixtures to {args.out_dir}")
    written: dict[str, Path] = {}
    for name, build in FIXTURES.items():
        try:
            obj = build(ctx)
        except ArcOutputMissing as exc:
            print(f"  {name:<20} skipped: {exc} (run `airbnb run-arc` first)")
            continue
        if obj is None:
            print(f"  {name:<20} skipped: source unavailable")
            continue
        written[name] = write_json(args.out_dir, name, obj)
    return written


if __name__ == "__main__":
    main()
