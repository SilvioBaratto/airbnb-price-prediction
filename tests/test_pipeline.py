"""The shared modeling plumbing: loading, the split, preprocessing and the ensemble helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from airbnb.domain import config
from airbnb.modeling import pipeline


def test_load_listings_keeps_features_and_target_in_contract_order(snapshot_csv: Path) -> None:
    """The id column is dropped on load, and the categoricals come back as strings."""
    frame = pipeline.load_listings(snapshot_csv)
    assert list(frame.columns) == config.FEATURES + [config.TARGET]
    assert frame["room_type"].map(type).eq(str).all()


def test_make_split_is_deterministic_and_disjoint(listings: pd.DataFrame) -> None:
    """Every part sees the same test rows, and no row is in both halves."""
    a = pipeline.make_split(listings)
    b = pipeline.make_split(listings)
    assert a.X_test.index.equals(b.X_test.index)
    assert not set(a.X_train.index) & set(a.X_test.index)
    assert len(a.X_test) == round(0.2 * len(listings))


def test_preprocessor_imputes_missing_values_and_ignores_unknown_categories(
    split: pipeline.Split,
) -> None:
    """A row with NaNs and a never-seen neighbourhood still gets a finite prediction."""
    model = pipeline.tree_pipeline(max_depth=3).fit(split.X_train, split.y_train)
    row = split.X_test.iloc[:1].copy()
    row[config.NUMERIC_FEATURES[2:]] = np.nan
    row["neighbourhood"] = "XV Nowhere"
    assert np.isfinite(model.predict(row)).all()


def test_column_groups_partition_the_design(split: pipeline.Split) -> None:
    """Every raw feature owns its design columns, and together they cover the design exactly."""
    model = pipeline.tree_pipeline(max_depth=2).fit(split.X_train, split.y_train)
    groups = pipeline.column_groups(model)
    assert set(groups) == set(config.FEATURES)
    indices = sorted(i for idx in groups.values() for i in idx)
    assert indices == list(range(pipeline.design(model, split.X_test).shape[1]))
    assert len(groups["room_type"]) == split.X_train["room_type"].nunique()


def test_design_names_spell_out_the_category(split: pipeline.Split) -> None:
    """One-hot columns read as ``column=value`` in printed tree rules."""
    model = pipeline.tree_pipeline(max_depth=2).fit(split.X_train, split.y_train)
    names = pipeline.design_names(model)
    assert "accommodates" in names
    assert "room_type=Private room" in names


def test_tree_predictions_average_to_the_ensemble_prediction(split: pipeline.Split) -> None:
    """Averaging the per-tree matrix reproduces what bagging and the forest predict."""
    for model in (
        pipeline.bagging_pipeline(n_estimators=8),
        pipeline.forest_pipeline(n_estimators=8),
    ):
        model.fit(split.X_train, split.y_train)
        per_tree = pipeline.tree_predictions(model, split.X_test)
        assert per_tree.shape == (8, len(split.X_test))
        np.testing.assert_allclose(per_tree.mean(axis=0), model.predict(split.X_test), rtol=1e-6)


def test_out_of_bag_predict_matches_scikit_learn(split: pipeline.Split) -> None:
    """The hand-rolled OOB average is the same number scikit-learn reports."""
    model = pipeline.forest_pipeline(n_estimators=30, oob_score=True)
    model.fit(split.X_train, split.y_train)
    pred, n_trees = pipeline.out_of_bag_predict(model, pipeline.design(model, split.X_train))
    graded = n_trees > 0
    reference = model.named_steps["est"].oob_prediction_
    np.testing.assert_allclose(pred[graded], reference[graded], rtol=1e-6)
    assert 0.25 < n_trees.mean() / 30 < 0.5


def test_boosting_quantile_pipeline_passes_the_quantile(split: pipeline.Split) -> None:
    """A 0.9-quantile booster prices above a 0.1-quantile one on average."""
    high = pipeline.boosting_pipeline(loss="quantile", quantile=0.9, max_iter=50)
    low = pipeline.boosting_pipeline(loss="quantile", quantile=0.1, max_iter=50)
    high.fit(split.X_train, split.y_train)
    low.fit(split.X_train, split.y_train)
    assert high.predict(split.X_test).mean() > low.predict(split.X_test).mean()


def test_format_report_quotes_the_baseline_and_dashes_missing_metrics() -> None:
    """Part 0 is the headline number, and parts without a metric print a dash."""
    report = pipeline.format_report(
        [
            pipeline.PartResult(1, "tree", 60.0, 61.0, 0.4),
            pipeline.PartResult(0, "mean", 86.0, 87.13, 0.0),
            pipeline.PartResult(6, "importance", None, None, None, note="km_to_center"),
        ]
    )
    lines = report.splitlines()
    assert "87.13" in lines[0]
    assert lines[3].lstrip().startswith("0 |")
    assert " - " in lines[-1] and "km_to_center" in lines[-1]
