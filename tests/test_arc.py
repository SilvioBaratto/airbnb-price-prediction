"""The eight parts, run on the small synthetic snapshot with small ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airbnb.modeling import arc, pipeline


@pytest.fixture
def args(tmp_path: Path) -> argparse.Namespace:
    """Arc options writing to a temporary output directory, without charts."""
    return argparse.Namespace(output_dir=tmp_path, charts=False, seed=7)


@pytest.fixture
def bagged(split: pipeline.Split) -> object:
    """A small bagging ensemble, standing in for Part 3's."""
    return pipeline.bagging_pipeline(n_estimators=20).fit(split.X_train, split.y_train)


@pytest.fixture
def forest(split: pipeline.Split) -> object:
    """A small forest fitted with OOB scores, standing in for Part 4's."""
    model = pipeline.forest_pipeline(n_estimators=40, oob_score=True)
    return model.fit(split.X_train, split.y_train)


def test_part1_tree_beats_the_mean(split: pipeline.Split, capsys: pytest.CaptureFixture) -> None:
    """Three levels of cuts already halve the error of always answering the mean."""
    baseline, tree = arc.run_part1(split)
    assert (baseline.part, tree.part) == (0, 1)
    assert tree.mae_test < baseline.mae_test
    assert "first cut:" in capsys.readouterr().out


def test_part2_full_tree_memorizes_and_cv_prunes_it(
    split: pipeline.Split, args: argparse.Namespace
) -> None:
    """The unpruned tree scores 0 on its own rows; the pruned one has fewer leaves."""
    result = arc.run_part2(split, args, grid_size=5)
    curve = pd.read_csv(args.output_dir / "part2_pruning.csv")
    assert "train 0.0" in result.note
    assert {"alpha", "leaves", "cv_mae", "train_mae", "test_mae"} <= set(curve.columns)
    assert curve["leaves"].is_monotonic_decreasing
    assert np.isfinite(result.mae_test)


def test_part3_bootstrap_holds_about_63_percent_and_averaging_helps(
    split: pipeline.Split, args: argparse.Namespace, capsys: pytest.CaptureFixture
) -> None:
    """Each resample holds ~1 - 1/e of the rows, and the average beats a lone tree."""
    result, bagged = arc.run_part3(split, args, n_estimators=20)
    out = capsys.readouterr().out
    share = float(out.split("holds ")[1].split("%")[0]) / 100
    assert share == pytest.approx(1 - np.exp(-1), abs=0.05)
    lone = float(result.note.split(": ")[1])
    assert result.mae_test < lone
    curve = pd.read_csv(args.output_dir / "part3_bagging.csv")
    assert curve["trees"].max() == 20
    assert len(bagged.named_steps["est"].estimators_) == 20


def test_part4_forest_varies_its_first_cut(
    split: pipeline.Split, args: argparse.Namespace, bagged: object
) -> None:
    """The forest returns an OOB-scored model and sweeps m up to all columns."""
    result, forest = arc.run_part4(split, args, bagged, n_estimators=30, sweep_trees=5)
    curve = pd.read_csv(args.output_dir / "part4_mtry.csv")
    p = pipeline.design(forest, split.X_test).shape[1]
    assert curve["m"].max() == p and curve["m"].min() == 1
    assert hasattr(forest.named_steps["est"], "oob_prediction_")
    assert "tree correlation" in result.note


def test_tree_correlation_is_lower_for_the_forest_than_for_bagging(
    split: pipeline.Split,
) -> None:
    """Hiding columns decorrelates the trees: the forest's rho is below bagging's."""
    bag = lambda n, s: pipeline.bagging_pipeline(n_estimators=n, seed=s)  # noqa: E731
    rf = lambda n, s: pipeline.forest_pipeline(n_estimators=n, max_features=2, seed=s)  # noqa: E731
    bagged = arc.tree_correlation(split, bag, n_sets=3, trees=15, repeats=2)
    forest = arc.tree_correlation(split, rf, n_sets=3, trees=15, repeats=2)
    for corr in (bagged, forest):
        assert 0.0 <= corr.rho <= 1.0 and corr.rho_sd >= 0.0 and corr.sigma2 > 0
        assert corr.slice_rows == len(split.y_train) // 3
    assert forest.rho < bagged.rho


@pytest.mark.parametrize("true_rho", [0.05, 0.3, 0.8])
def test_intraclass_correlation_recovers_a_known_rho(true_rho: float) -> None:
    """On data built with a known shared share, the estimate lands on it, with five sets.

    The plain-variance estimator this replaced read, over 20 such draws, 0.66 of the truth at
    rho = 0.05 (the arc's regime) and 0.84 at rho = 0.3.
    """
    rng = np.random.default_rng(0)
    sets, trees, listings, sigma2 = 5, 25, 4000, 100.0
    shared = rng.normal(0, np.sqrt(true_rho * sigma2), (sets, 1, listings))
    own = rng.normal(0, np.sqrt((1 - true_rho) * sigma2), (sets, trees, listings))
    rho, total = arc.intraclass_correlation(shared + own)
    assert rho == pytest.approx(true_rho, abs=0.02)
    assert total == pytest.approx(sigma2, rel=0.05)


def test_intraclass_correlation_is_never_negative() -> None:
    """Identical sets (no shared quirk at all) give rho = 0, not a negative correlation."""
    own = np.random.default_rng(1).normal(size=(4, 10, 500))
    rho, _ = arc.intraclass_correlation(own - own.mean(axis=1, keepdims=True))
    assert rho == 0.0


def test_part5_oob_error_is_close_to_the_test_error(
    split: pipeline.Split, forest: object, capsys: pytest.CaptureFixture
) -> None:
    """The free OOB estimate lands near the held-out error."""
    result = arc.run_part5(split, forest)
    oob = float(result.note.split()[2].rstrip(","))
    assert oob == pytest.approx(result.mae_test, rel=0.3)
    assert "matches sklearn" in capsys.readouterr().out


def test_part6_ranks_every_column_and_finds_the_real_drivers(
    split: pipeline.Split, forest: object, args: argparse.Namespace
) -> None:
    """Guests and bathrooms drive the synthetic price; shuffling them hurts most."""
    arc.run_part6(split, forest, args)
    table = pd.read_csv(args.output_dir / "part6_importance.csv")
    assert set(table["column"]) == set(pipeline.FEATURES)
    assert table["column"].iloc[0] in {"accommodates", "beds", "bathrooms", "bedrooms"}
    assert table["split_gain_share"].sum() == pytest.approx(1.0)


def test_part7_tunes_rounds_on_validation_for_both_losses(
    split: pipeline.Split, args: argparse.Namespace, capsys: pytest.CaptureFixture
) -> None:
    """Two results (squared and absolute loss), each with a round count within the cap."""
    assert len(arc.run_part7(split, args, max_iter=5)) == 2  # fewer rounds than the milestones
    for line in capsys.readouterr().out.splitlines():
        if "by round:" in line:
            rounds = [int(chunk.split(" ->")[0]) for chunk in line.split(": ", 1)[1].split(", ")]
            assert rounds == sorted(set(rounds))  # each round once, in order
    results = arc.run_part7(split, args, max_iter=120)
    assert [r.part for r in results] == [7, 7]
    for r in results:
        rounds = int(r.name.split("M=")[1].rstrip(")"))
        assert 1 <= rounds <= 120
    curve = pd.read_csv(args.output_dir / "part7_boosting.csv")
    assert len(curve) == 120
    assert {"val_mae_squared", "test_mae_absolute"} <= set(curve.columns)


def test_part8_ranks_the_lineup_on_both_datasets(
    listings: pd.DataFrame, args: argparse.Namespace
) -> None:
    """The lineup is ranked on both datasets, and OLS wins on the near-linear diabetes data."""
    result = arc.run_part8(listings, args)
    board = pd.read_csv(args.output_dir / "part8_nfl.csv")
    assert set(board["dataset"]) == {"airbnb_rome", "diabetes"}
    assert board.groupby("dataset")["model"].count().eq(len(arc.lineup([], []))).all()
    diabetes = board[board["dataset"] == "diabetes"]
    assert diabetes.iloc[0]["model"] in {"OLS", "Lasso"}
    assert "Rome #1" in result.note and "CV MAE" in result.note
    assert result.mae_test is None  # a CV score is not a held-out test score


def test_main_runs_the_whole_arc_on_a_snapshot_file(snapshot_csv: Path, tmp_path: Path) -> None:
    """The orchestrator wires every part and prints the summary table."""
    out = tmp_path / "o"
    results = arc.main(["--listings", str(snapshot_csv), "--output-dir", str(out)])
    assert [r.part for r in results] == [0, 1, 2, 3, 4, 5, 6, 7, 7, 8]
    summary = pd.read_csv(out / "arc_summary.csv")
    assert summary["part"].tolist() == [r.part for r in results]
    manifest = json.loads((out / "arc_run.json").read_text())
    assert manifest["n_listings"] == 400 and manifest["test_size"] == 0.2
    assert manifest["snapshot_sha256"] == pipeline.file_sha256(snapshot_csv)
    assert manifest["listings"] == snapshot_csv.name
    written = sorted(p.name for p in out.glob("*.csv"))
    assert sorted(manifest["outputs"]) == written and "arc_summary.csv" in written
    for name, digest in manifest["outputs"].items():
        assert pipeline.file_sha256(out / name) == digest


def test_a_crashed_run_leaves_no_manifest(
    snapshot_csv: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The old manifest goes first, so CSVs from a half-finished run never look complete."""
    out = tmp_path / "o"
    out.mkdir()
    (out / "arc_run.json").write_text("{}")

    def boom(*_args, **_kwargs):
        raise RuntimeError("killed mid-run")

    monkeypatch.setattr(arc, "run_part2", boom)
    with pytest.raises(RuntimeError):
        arc.main(["--listings", str(snapshot_csv), "--output-dir", str(out)])
    assert not (out / "arc_run.json").exists()
