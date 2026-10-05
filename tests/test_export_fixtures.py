"""The videocraft fixture export, end to end on the synthetic snapshot.

Runs the real arc into a temporary directory once, then exports from it. The fixtures must agree
with the arc's own numbers and with models refitted independently here, and the export must
refuse an arc run it could not reproduce.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from airbnb.domain import config
from airbnb.modeling import arc, pipeline
from tests.factories import make_listings

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_fixtures.py"


@functools.cache
def _load_script() -> ModuleType:
    """Import ``scripts/export_fixtures.py``, which is a script, not a package module."""
    spec = importlib.util.spec_from_file_location("export_fixtures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so the module must be registered
    # before its body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Export(NamedTuple):
    """One arc run and the export made from it."""

    snapshot: Path
    arc_dir: Path
    out_dir: Path
    written: dict[str, Path]


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Export:
    """Run the arc and the export once for the whole module."""
    root = tmp_path_factory.mktemp("fixtures")
    snapshot = root / "listings.csv"
    make_listings().to_csv(snapshot, index=False)
    arc_dir = root / "arc"
    arc.main(["--listings", str(snapshot), "--output-dir", str(arc_dir)])
    out_dir = root / "out"
    written = _load_script().main(
        ["--listings", str(snapshot), "--arc-dir", str(arc_dir), "--out-dir", str(out_dir)]
    )
    return Export(snapshot, arc_dir, out_dir, written)


def _read(export: Export, name: str) -> dict:
    """Parse a fixture strictly: NaN or Infinity anywhere fails, as it would in a browser."""

    def _reject(token: str) -> None:
        raise ValueError(f"{name} contains {token}, which is not JSON")

    return json.loads((export.out_dir / name).read_text(), parse_constant=_reject)


def _summary_row(export: Export, part: int, contains: str = "") -> pd.Series:
    """The one arc summary row of ``part`` whose name contains ``contains``."""
    summary = pd.read_csv(export.arc_dir / "arc_summary.csv")
    rows = summary[(summary["part"] == part) & summary["name"].str.contains(contains, regex=False)]
    assert len(rows) == 1
    return rows.iloc[0]


def _context(export: Export, **overrides):
    """A ``Contesto`` over the module's snapshot and arc run, with fields replaced as given."""
    module = _load_script()
    listings = pipeline.load_listings(export.snapshot)
    test_size = overrides.pop("test_size", 0.2)
    fields = {
        "listings": listings,
        "split": pipeline.make_split(listings, test_size=test_size),
        "seed": config.SEED,
        "test_size": test_size,
        "arc_dir": export.arc_dir,
        "listings_path": export.snapshot,
        "places_path": Path(module.paths.PLACES_CSV),
        **overrides,
    }
    return module.Contesto(**fields)


def test_every_fixture_is_valid_json_with_its_provenance(exported: Export) -> None:
    """All twelve fixtures exist, parse strictly, and name the exact data and split behind them."""
    assert set(exported.written) == set(_load_script().FIXTURES)
    digest = pipeline.file_sha256(exported.snapshot)
    for name in exported.written:
        payload = _read(exported, name)
        if name != "repo.json":
            fonte = payload["fonte"]
            assert fonte["seme"] == config.SEED and fonte["testSize"] == 0.2
            assert fonte["snapshot"] == digest[:12] and fonte["nAnnunci"] == 400


def test_fixtures_quote_the_arc_numbers_to_the_cent(exported: Export) -> None:
    """Every number shared with the arc's summary table is the arc's, not a lookalike."""

    def _mae(part: int, contains: str = "") -> float:
        return round(float(_summary_row(exported, part, contains)["mae_test"]), 2)

    albero = _read(exported, "albero.json")
    assert albero["livelli"][0]["maeTest"] == _mae(0)
    assert albero["livelli"][3]["maeTest"] == _mae(1)
    assert _read(exported, "potatura.json")["scelto"]["maeTest"] == _mae(2)
    assert _read(exported, "bagging.json")["maeTest"] == _mae(3)
    assert _read(exported, "foresta.json")["maeTest"] == {"bagging": _mae(3), "foresta": _mae(4)}
    assert _read(exported, "oob.json")["maeTest"] == _mae(4)
    best = _read(exported, "boosting.json")["migliori"]
    for key, loss in (("quadratica", "squared"), ("assoluta", "absolute")):
        row = _summary_row(exported, 7, f"{loss} loss")
        assert best[key]["maeTest"] == round(float(row["mae_test"]), 2)
        rounds = re.search(r"M=(\d+)", row["name"])
        assert rounds is not None and best[key]["alberi"] == int(rounds.group(1))


def test_tree_correlation_matches_what_the_arc_reported(exported: Export) -> None:
    """The exporter's rho is the arc's rho, within the note's two-decimal rounding.

    The fixture ships three decimals, so rounding it again to two could land on the other side
    of a half (0.1853 -> 0.185 -> 0.18): compare within half a hundredth instead.
    """
    note = _summary_row(exported, 4)["note"]
    bagging, forest = (float(x) for x in re.findall(r"\d+\.\d+", note))
    corr = _read(exported, "foresta.json")["correlazione"]
    assert corr["bagging"]["rho"] == pytest.approx(bagging, abs=0.0051)
    assert corr["foresta"]["rho"] == pytest.approx(forest, abs=0.0051)
    for block in (corr["bagging"], corr["foresta"]):
        assert 0 <= block["rho"] <= 1 and block["rhoSd"] >= 0
        assert block["pavimento"] <= block["varianzaMedia"]


def test_pruning_curve_keeps_every_alpha_distinct_and_positive(exported: Export) -> None:
    """The path spans twenty decades: fixed decimals would flatten its start to zero."""
    curve = _read(exported, "potatura.json")["curva"]
    alphas = [p["alpha"] for p in curve]
    csv = pd.read_csv(exported.arc_dir / "part2_pruning.csv")
    assert len(alphas) == len(csv)
    assert all(a > 0 for a in alphas)
    assert alphas == sorted(alphas) and len(set(alphas)) == len(alphas)
    np.testing.assert_allclose(alphas, csv["alpha"], rtol=1e-3)


def test_tree_rows_walk_the_tree_the_way_the_model_did(exported: Export) -> None:
    """At every node the shipped value and threshold send the row the way its path goes."""
    albero = _read(exported, "albero.json")
    nodes = {n["id"]: n for n in albero["nodi"]}
    for row in albero["righe"]:
        path = row["percorso"]
        assert path[0] == 0
        for here, there in zip(path, path[1:]):
            node = nodes[here]
            goes_left = row["valori"][node["colonna"]] <= node["soglia"]
            assert there == (node["sinistra"] if goes_left else node["destra"])
        leaf = nodes[path[-1]]
        assert leaf["colonna"] is None and leaf["prezzoMedio"] == row["stima"]


def test_tree_nodes_are_the_refitted_trees_nodes(exported: Export) -> None:
    """Every shipped cut (column, threshold, size, mean) equals an independent refit's.

    The walk above only notices a threshold error some sampled row falls into; comparing node
    by node catches a threshold that is off by any amount at the shipped precision.
    """
    listings = pipeline.load_listings(exported.snapshot)
    split = pipeline.make_split(listings)
    model = pipeline.tree_pipeline(max_depth=3).fit(split.X_train, split.y_train)
    tree = model.named_steps["est"].tree_
    names = pipeline.design_names(model)
    decimals = _load_script().TREE_DECIMALS

    nodes = _read(exported, "albero.json")["nodi"]
    assert len(nodes) == tree.node_count
    for node in nodes:
        i = node["id"]
        assert node["annunci"] == int(tree.n_node_samples[i])
        assert node["prezzoMedio"] == round(float(tree.value[i].ravel()[0]), 2)
        if tree.children_left[i] == -1:
            assert node["colonna"] is None and node["soglia"] is None
        else:
            assert node["colonna"] == names[tree.feature[i]]
            assert node["soglia"] == round(float(tree.threshold[i]), decimals)


def test_boosting_residuals_start_from_the_mean_and_end_at_the_chosen_round(
    exported: Export,
) -> None:
    """Round 0 is the training mean; the last round shown is the one validation chose."""
    boosting = _read(exported, "boosting.json")
    best = boosting["migliori"]["quadratica"]["alberi"]
    for listing in boosting["residui"]:
        rounds = [s["alberi"] for s in listing["stime"]]
        assert rounds[0] == 0 and rounds[-1] == best
        assert rounds == sorted(rounds)


def test_oob_fixture_matches_an_independently_refitted_forest(exported: Export) -> None:
    """Draw counts, the never-saw-it total and the OOB verdict agree with a fresh forest."""
    listings = pipeline.load_listings(exported.snapshot)
    split = pipeline.make_split(listings)
    forest = pipeline.forest_pipeline(oob_score=True).fit(split.X_train, split.y_train)
    draws = [int(np.sum(bag == 0)) for bag in forest.named_steps["est"].estimators_samples_]
    pred, _ = pipeline.out_of_bag_predict(forest, pipeline.design(forest, split.X_train))

    oob = _read(exported, "oob.json")
    assert oob["estrazioniPerAlbero"] == draws[: len(oob["estrazioniPerAlbero"])]
    assert oob["alberiCheNonLoHannoVisto"] == sum(d == 0 for d in draws)
    assert oob["stimaOob"] == round(float(pred[0]), 2)


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 1},
        {"test_size": 0.3},
        {"snapshot_sha256": "0" * 64},
        {"bagged_trees": 7},
        {"forest_trees": 7},
        {"snapshot_sha256": None},
    ],
    ids=["seed", "test_size", "snapshot", "bagged_trees", "forest_trees", "older_arc"],
)
def test_a_manifest_from_another_run_is_refused(
    exported: Export, tmp_path: Path, change: dict
) -> None:
    """Any recorded setting that differs from the export's, or is missing, refuses the run."""
    stale = tmp_path / "arc"
    stale.mkdir()
    for f in exported.arc_dir.iterdir():
        (stale / f.name).write_bytes(f.read_bytes())
    run = json.loads((stale / "arc_run.json").read_text())
    run.update(change)
    run = {k: v for k, v in run.items() if v is not None}
    (stale / "arc_run.json").write_text(json.dumps(run))

    ctx = _context(exported, arc_dir=stale)
    with pytest.raises(_load_script().ArcOutputMissing, match="another run"):
        ctx.arc_csv("arc_summary.csv")


@pytest.mark.parametrize("tamper", ["edit", "unrecorded"])
def test_a_csv_that_is_not_the_runs_own_is_refused(
    exported: Export, tmp_path: Path, tamper: str
) -> None:
    """A CSV edited after the run, or one the manifest never recorded, is not quoted."""
    arc_dir = tmp_path / "arc"
    arc_dir.mkdir()
    for f in exported.arc_dir.iterdir():
        (arc_dir / f.name).write_bytes(f.read_bytes())
    if tamper == "edit":
        csv = arc_dir / "part3_bagging.csv"
        csv.write_text(csv.read_text().replace("\n1,", "\n1,1", 1))
        expected = "changed after the run"
    else:
        run = json.loads((arc_dir / "arc_run.json").read_text())
        del run["outputs"]["part3_bagging.csv"]
        (arc_dir / "arc_run.json").write_text(json.dumps(run))
        expected = "not among the run's recorded outputs"

    ctx = _context(exported, arc_dir=arc_dir)
    with pytest.raises(_load_script().ArcOutputMissing, match=expected):
        ctx.arc_csv("part3_bagging.csv")
    assert ctx.arc_csv("part2_pruning.csv") is not None  # the untouched CSVs still pass


def test_an_export_with_another_split_is_refused(exported: Export) -> None:
    """Exporting with --test-size 0.3 against a 0.2 run would mix two splits: refused."""
    ctx = _context(exported, test_size=0.3)
    with pytest.raises(_load_script().ArcOutputMissing, match="test_size"):
        ctx.arc_mae(0)


def test_without_a_manifest_only_the_arc_free_fixtures_are_written(
    exported: Export, tmp_path: Path
) -> None:
    """A crashed or absent arc run blocks every fixture that quotes it, and only those."""
    partial = tmp_path / "arc"
    partial.mkdir()
    for f in exported.arc_dir.iterdir():
        if f.name != "arc_run.json":
            (partial / f.name).write_bytes(f.read_bytes())
    written = _load_script().main(
        [
            "--listings",
            str(exported.snapshot),
            "--arc-dir",
            str(partial),
            "--out-dir",
            str(tmp_path / "o"),
        ]
    )
    assert set(written) == {"repo.json", "mappa.json", "albero.json", "simulatore.json"}


def test_arc_mae_refuses_an_ambiguous_row(exported: Export) -> None:
    """Part 7 has two rows; asking for it without naming one is an error, not the first row."""
    with pytest.raises(ValueError, match="2 arc rows"):
        _context(exported).arc_mae(7)
