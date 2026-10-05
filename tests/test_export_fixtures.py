"""The videocraft fixture export, end to end on the synthetic snapshot.

Runs the real arc into a temporary directory once, then exports from it: the fixtures must agree
with the arc's own numbers, and refuse an arc run that belongs to another seed.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from airbnb.modeling import arc
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


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict[str, Path]]:
    """Run the arc and the export once: ``(arc_dir, out_dir, written)``."""
    root = tmp_path_factory.mktemp("fixtures")
    snapshot = root / "listings.csv"
    make_listings().to_csv(snapshot, index=False)
    arc_dir = root / "arc"
    arc.main(["--listings", str(snapshot), "--output-dir", str(arc_dir)])
    out_dir = root / "out"
    written = _load_script().main(
        ["--listings", str(snapshot), "--arc-dir", str(arc_dir), "--out-dir", str(out_dir)]
    )
    return arc_dir, out_dir, written


def _read(out_dir: Path, name: str) -> dict:
    return json.loads((out_dir / name).read_text())


def test_every_fixture_is_written_with_its_provenance(exported) -> None:
    """All twelve fixtures exist, and each data fixture says where its numbers come from."""
    _, out_dir, written = exported
    assert set(written) == set(_load_script().FIXTURES)
    for name in written:
        payload = _read(out_dir, name)
        if name != "repo.json":
            assert payload["fonte"]["seme"] == 20260907
            assert payload["fonte"]["nAnnunci"] == 400


def test_fixtures_quote_the_arc_numbers_not_their_own(exported) -> None:
    """Numbers shared with the arc match its summary table to the cent."""
    arc_dir, out_dir, _ = exported
    summary = pd.read_csv(arc_dir / "arc_summary.csv")

    def _arc_mae(part: int) -> float:
        return round(float(summary.loc[summary["part"] == part, "mae_test"].iloc[0]), 2)

    albero = _read(out_dir, "albero.json")
    assert albero["livelli"][3]["maeTest"] == _arc_mae(1)
    assert albero["livelli"][0]["maeTest"] == _arc_mae(0)
    assert _read(out_dir, "bagging.json")["maeTest"] == _arc_mae(3)
    assert _read(out_dir, "foresta.json")["maeTest"]["foresta"] == _arc_mae(4)
    assert _read(out_dir, "potatura.json")["scelto"]["maeTest"] == _arc_mae(2)


def test_tree_fixture_is_a_walkable_tree(exported) -> None:
    """Every row's path starts at the root and ends on a leaf whose mean is the row's estimate."""
    _, out_dir, _ = exported
    albero = _read(out_dir, "albero.json")
    nodes = {n["id"]: n for n in albero["nodi"]}
    for row in albero["righe"]:
        path = row["percorso"]
        assert path[0] == 0
        leaf = nodes[path[-1]]
        assert leaf["colonna"] is None and leaf["prezzoMedio"] == row["stima"]


def test_boosting_residuals_start_from_the_mean_and_end_at_the_chosen_round(exported) -> None:
    """Round 0 is the training mean; the last round shown is the one validation chose."""
    _, out_dir, _ = exported
    boosting = _read(out_dir, "boosting.json")
    best = boosting["migliori"]["quadratica"]["alberi"]
    for listing in boosting["residui"]:
        rounds = [s["alberi"] for s in listing["stime"]]
        assert rounds[0] == 0 and rounds[-1] == best
        assert rounds == sorted(rounds)


def test_oob_counts_match_the_trees_that_never_saw_the_listing(exported) -> None:
    """The per-tree draw counts shown agree with the never-saw-it total over the first trees."""
    _, out_dir, _ = exported
    oob = _read(out_dir, "oob.json")
    draws = oob["estrazioniPerAlbero"]
    assert all(d >= 0 for d in draws)
    assert sum(d == 0 for d in draws) <= oob["alberiCheNonLoHannoVisto"]


def test_a_stale_arc_run_is_refused(exported, tmp_path: Path) -> None:
    """Fixtures that quote the arc are skipped when its run used another seed."""
    arc_dir, _, _ = exported
    stale = tmp_path / "arc"
    stale.mkdir()
    for f in arc_dir.iterdir():
        (stale / f.name).write_bytes(f.read_bytes())
    run = json.loads((stale / "arc_run.json").read_text())
    (stale / "arc_run.json").write_text(json.dumps({**run, "seed": 1}))

    snapshot = tmp_path / "listings.csv"
    make_listings().to_csv(snapshot, index=False)
    written = _load_script().main(
        ["--listings", str(snapshot), "--arc-dir", str(stale), "--out-dir", str(tmp_path / "o")]
    )
    assert "albero.json" in written and "mappa.json" in written
    assert "arc_metrics.json" not in written and "boosting.json" not in written
