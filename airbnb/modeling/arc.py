"""Orchestrator for the eight-part tree-ensemble arc (README "The model, part by part").

Loads the snapshot **once**, carves the one fixed 80/20 split (Parts 1-7 share it), then runs
Parts 1->8 as ``run_partN`` section functions and prints the summary table. Every number is
produced here by running scikit-learn at ``config.SEED`` — never hand-written. Whenever a part
has to *choose* something (pruning strength, number of boosting rounds) it chooses on the
training rows; the test rows are only ever scored. Part 8 runs its own 5-fold CV on the full
snapshot and on a second, unrelated dataset.

Usage::

    python -m airbnb.modeling.arc                  # the committed data/raw/listings_rome.csv
    python -m airbnb.modeling.arc --charts         # also write the PNG charts to output/
    airbnb run-arc --seed 7 --output-dir /tmp/arc  # same flags through the unified CLI
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_text

from airbnb.domain import config
from airbnb.infrastructure import paths
from airbnb.modeling import pipeline as modeling

PRUNING_GRID_SIZE = 20
BAGGING_B_GRID = (1, 2, 3, 5, 10, 25, 50, 100, 200)
FIRST_SPLIT_SAMPLE = 50
M_SWEEP_TREES = 100
CORRELATION_SETS = 5
CORRELATION_TREES = 25
BOOSTING_MAX_ITER = 3000
CV_FOLDS = 5


# --- CLI --------------------------------------------------------------------
def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the arc flags."""
    parser = argparse.ArgumentParser(
        prog="airbnb run-arc", description="Run the eight-part tree-ensemble arc."
    )
    parser.add_argument(
        "--listings",
        type=Path,
        default=paths.LISTINGS_CSV,
        help="cleaned snapshot to learn from (default: data/raw/listings_rome.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=config.SEED, help=f"random_state (default: {config.SEED})"
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="held-out test fraction (default: 0.2)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.OUTPUT_DIR,
        help="where the per-part CSVs (+ --charts PNGs) are written (default: output/)",
    )
    parser.add_argument(
        "--charts", action="store_true", help="also write a PNG chart next to each curve CSV"
    )
    return parser.parse_args(argv)


def _owner(fitted: Pipeline) -> dict[int, str]:
    """Map each design column index back to the raw feature it came from."""
    return {i: col for col, idx in modeling.column_groups(fitted).items() for i in idx}


def _write(frame: pd.DataFrame, args: argparse.Namespace, name: str) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / name
    frame.to_csv(path, index=False)
    return path


# --- Part 1 — the tree that cuts the table ----------------------------------
def run_part1(split: modeling.Split, *, seed: int = config.SEED) -> list[modeling.PartResult]:
    """The mean baseline, the single best cut (a stump), and a depth-3 tree of 8 leaves.

    CART picks, among every column and every threshold, the cut that leaves the smallest sum of
    squared errors in the two halves, and then answers the mean price of each half. A tree is
    that cut repeated inside each half.
    """
    baseline = modeling.mean_pipeline().fit(split.X_train, split.y_train)
    base_train = modeling.mae(split.y_train, baseline.predict(split.X_train))
    base_test, base_r2 = modeling.score(baseline, split.X_test, split.y_test)

    stump = modeling.tree_pipeline(max_depth=1, seed=seed).fit(split.X_train, split.y_train)
    names = modeling.design_names(stump)
    root = stump.named_steps["est"].tree_
    cut = names[root.feature[0]]
    left_mean, right_mean = float(root.value[1].ravel()[0]), float(root.value[2].ravel()[0])
    n_left, n_right = int(root.n_node_samples[1]), int(root.n_node_samples[2])
    stump_test, _ = modeling.score(stump, split.X_test, split.y_test)

    tree = modeling.tree_pipeline(max_depth=3, seed=seed).fit(split.X_train, split.y_train)
    tree_train = modeling.mae(split.y_train, tree.predict(split.X_train))
    tree_test, tree_r2 = modeling.score(tree, split.X_test, split.y_test)
    rules = export_text(tree.named_steps["est"], feature_names=names, decimals=1)

    print(f"[Part 1] mean baseline: always {baseline.predict(split.X_test[:1])[0]:.2f} EUR")
    print(f"[Part 1]   test MAE={base_test:.2f}")
    print(
        f"[Part 1] first cut: {cut} <= {root.threshold[0]:.2f} -> "
        f"{n_left:,} listings at {left_mean:.0f} EUR | {n_right:,} at {right_mean:.0f} EUR"
    )
    print(f"[Part 1]   one cut alone: test MAE={stump_test:.2f}")
    print(f"[Part 1] depth-3 tree (8 leaves): train MAE={tree_train:.2f}  test MAE={tree_test:.2f}")
    for line in rules.rstrip().splitlines():
        print(f"[Part 1]   {line}")
    return [
        modeling.PartResult(
            0, "mean (no model)", base_train, base_test, base_r2, note="the number to beat"
        ),
        modeling.PartResult(
            1,
            "tree, depth 3 (8 leaves)",
            tree_train,
            tree_test,
            tree_r2,
            note=f"first cut: {cut}",
        ),
    ]


# --- Part 2 — when to stop cutting -------------------------------------------
def run_part2(
    split: modeling.Split,
    args: argparse.Namespace,
    *,
    grid_size: int = PRUNING_GRID_SIZE,
    seed: int = config.SEED,
) -> modeling.PartResult:
    """Grow the tree until it memorizes, then prune it back by cost-complexity.

    ``C_alpha(T) = error(T) + alpha * |T|``: every leaf costs ``alpha``. The pruning path gives
    the nested sequence of subtrees (weakest link first); ``alpha`` is chosen by 5-fold CV on
    the training rows. Writes ``part2_pruning.csv`` (alpha, leaves, CV/train/test MAE).
    """
    full = modeling.tree_pipeline(seed=seed).fit(split.X_train, split.y_train)
    full_leaves = int(full.named_steps["est"].get_n_leaves())
    full_train = modeling.mae(split.y_train, full.predict(split.X_train))
    full_test, _ = modeling.score(full, split.X_test, split.y_test)

    Xd = modeling.design(full, split.X_train)
    pruning = DecisionTreeRegressor(random_state=seed).cost_complexity_pruning_path(
        Xd, split.y_train
    )
    # Drop alpha=0 (the full tree) and the last alpha (the root alone); span the rest evenly in
    # log scale, since the path's alphas range over many orders of magnitude.
    alphas = pruning.ccp_alphas[1:-1]
    alphas = alphas[alphas > 0]
    grid = np.unique(np.geomspace(alphas.min(), alphas.max(), grid_size))

    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    rows: list[dict[str, float]] = []
    for alpha in grid:
        model = modeling.tree_pipeline(ccp_alpha=float(alpha), seed=seed)
        cv = -cross_val_score(
            model,
            split.X_train,
            split.y_train,
            cv=folds,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        ).mean()
        model.fit(split.X_train, split.y_train)
        rows.append(
            {
                "alpha": float(alpha),
                "leaves": float(model.named_steps["est"].get_n_leaves()),
                "cv_mae": float(cv),
                "train_mae": modeling.mae(split.y_train, model.predict(split.X_train)),
                "test_mae": modeling.mae(split.y_test, model.predict(split.X_test)),
            }
        )
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["cv_mae"].idxmin()]
    pruned = modeling.tree_pipeline(ccp_alpha=float(best["alpha"]), seed=seed)
    pruned.fit(split.X_train, split.y_train)
    _, pruned_r2 = modeling.score(pruned, split.X_test, split.y_test)
    path = _write(curve, args, "part2_pruning.csv")
    if args.charts:
        _plot_lines(
            curve,
            "leaves",
            {"CV MAE": "cv_mae", "train MAE": "train_mae", "test MAE": "test_mae"},
            args.output_dir / "part2_pruning.png",
            title="Part 2 — pruning: error vs number of leaves",
            xlabel="leaves",
            logx=True,
        )

    print(
        f"[Part 2] full tree: {full_leaves:,} leaves  train MAE={full_train:.2f} "
        f"test MAE={full_test:.2f} (memorized)"
    )
    print(
        f"[Part 2] CV picks alpha={best['alpha']:.1f} -> {int(best['leaves']):,} leaves  "
        f"CV MAE={best['cv_mae']:.2f}  test MAE={best['test_mae']:.2f}"
    )
    print(f"[Part 2]   wrote {path}" + (" (+ PNG)" if args.charts else ""))
    return modeling.PartResult(
        2,
        f"pruned tree ({int(best['leaves'])} leaves)",
        float(best["train_mae"]),
        float(best["test_mae"]),
        pruned_r2,
        note=f"full tree: {full_leaves} leaves, train {full_train:.1f} / test {full_test:.1f}",
    )


# --- Part 3 — redo the table by drawing lots -----------------------------------
def run_part3(
    split: modeling.Split,
    args: argparse.Namespace,
    *,
    n_estimators: int = modeling.N_BAGGED_TREES,
    seed: int = config.SEED,
) -> tuple[modeling.PartResult, Pipeline]:
    """Bagging: B full trees on B bootstrap resamples, their predictions averaged.

    A bootstrap sample draws n rows *with replacement*, so ~63% of the rows appear (some
    twice) and ~37% are left out. Each lone tree is as wild as Part 2's full tree; their average
    is not. Writes ``part3_bagging.csv`` (B, test MAE of the first B trees averaged).
    """
    bagged = modeling.bagging_pipeline(n_estimators=n_estimators, seed=seed)
    bagged.fit(split.X_train, split.y_train)
    per_tree = modeling.tree_predictions(bagged, split.X_test)
    lone = float(np.mean([modeling.mae(split.y_test, p) for p in per_tree]))
    grid = [b for b in BAGGING_B_GRID if b < n_estimators] + [n_estimators]
    curve = pd.DataFrame(
        {
            "trees": grid,
            "test_mae": [modeling.mae(split.y_test, per_tree[:b].mean(0)) for b in grid],
        }
    )
    n_train = len(split.y_train)
    samples = bagged.named_steps["est"].estimators_samples_
    unique_share = float(np.mean([len(np.unique(s)) / n_train for s in samples]))

    train_mae = modeling.mae(split.y_train, bagged.predict(split.X_train))
    test_mae, test_r2 = modeling.score(bagged, split.X_test, split.y_test)
    path = _write(curve, args, "part3_bagging.csv")
    if args.charts:
        _plot_lines(
            curve,
            "trees",
            {"test MAE": "test_mae"},
            args.output_dir / "part3_bagging.png",
            title="Part 3 — bagging: error vs number of averaged trees",
            xlabel="trees averaged (B)",
            logx=True,
        )

    print(
        f"[Part 3] each bootstrap sample holds {unique_share:.1%} of the distinct listings "
        f"(1 - 1/e = {1 - np.exp(-1):.1%})"
    )
    print(f"[Part 3] a lone bootstrap tree: test MAE={lone:.2f} on average")
    print(
        f"[Part 3] {n_estimators} trees averaged: "
        f"train MAE={train_mae:.2f}  test MAE={test_mae:.2f}"
    )
    print(f"[Part 3]   wrote {path}" + (" (+ PNG)" if args.charts else ""))
    return (
        modeling.PartResult(
            3,
            f"bagging (B={n_estimators})",
            train_mae,
            test_mae,
            test_r2,
            note=f"a lone tree: {lone:.1f}",
        ),
        bagged,
    )


# --- Part 4 — trees that stop copying each other -------------------------------
def _first_cuts(fitted: Pipeline, owner: dict[int, str], n_trees: int) -> Counter[str]:
    """Count which raw column the first ``n_trees`` trees cut on at their root."""
    ensemble = fitted.named_steps["est"]
    feature_sets = getattr(ensemble, "estimators_features_", None)
    cuts: Counter[str] = Counter()
    for i, tree in enumerate(ensemble.estimators_[:n_trees]):
        local = int(tree.tree_.feature[0])
        column = int(feature_sets[i][local]) if feature_sets is not None else local
        cuts[owner[column]] += 1
    return cuts


def _tree_correlation(
    split: modeling.Split,
    make: Callable[[int, int], Pipeline],
    *,
    n_sets: int = CORRELATION_SETS,
    trees: int = CORRELATION_TREES,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Estimate ESL's ``(rho, sigma2)`` for an ensemble recipe, as in ESL figure 15.9.

    ``sigma2`` is how much one tree's prediction for a listing varies, ``rho`` the correlation
    between two trees grown on the *same* training data. Both are about repeating the whole
    experiment on fresh data, so the training rows are cut into ``n_sets`` disjoint slices, each
    standing in for an independent sample of the city, and ``trees`` trees are grown on each.
    Across slices the ensemble mean varies by ``rho*sigma2 + (1 - rho)*sigma2/trees``, which is
    solved for ``rho``. Correlating the trees' raw errors instead would mostly measure the noise
    in the prices, which every tree shares.

    Args:
        split: the shared split; slices come from its training rows, predictions are taken on
            its test listings.
        make: builds an unfitted ensemble from ``(n_estimators, seed)``.
        n_sets: how many disjoint training slices to grow ensembles on.
        trees: ensemble size per slice.
        seed: shuffles the training rows before slicing.

    Returns:
        ``(rho, sigma2)``, averaged over the test listings.
    """
    order = np.random.default_rng(seed).permutation(len(split.y_train))
    per_set = []
    for k, rows in enumerate(np.array_split(order, n_sets)):
        model = make(trees, seed + k).fit(split.X_train.iloc[rows], split.y_train[rows])
        per_set.append(modeling.tree_predictions(model, split.X_test))
    sigma2 = float(np.vstack(per_set).var(axis=0).mean())
    between = float(np.var([p.mean(axis=0) for p in per_set], axis=0).mean())
    rho = (between / sigma2 - 1 / trees) / (1 - 1 / trees)
    return float(rho), sigma2


def run_part4(
    split: modeling.Split,
    args: argparse.Namespace,
    bagged: Pipeline,
    *,
    n_estimators: int = modeling.N_FOREST_TREES,
    sweep_trees: int = M_SWEEP_TREES,
    seed: int = config.SEED,
) -> tuple[modeling.PartResult, Pipeline]:
    """Random forest: at each split only ``m`` random columns may be cut on.

    Bagged trees all open with the same dominant cut, so they make the same mistakes and
    averaging them cancels little. Hiding columns forces different first cuts, lowers the
    correlation ``rho`` between trees, and with it the variance of their average,
    ``rho*sigma2 + (1 - rho)*sigma2/B``, which no number of trees ``B`` can push below
    ``rho*sigma2``. The forest is fitted with ``oob_score=True`` so
    Parts 5 and 6 can reuse it. Writes ``part4_mtry.csv`` (m, test MAE).
    """
    forest = modeling.forest_pipeline(n_estimators=n_estimators, oob_score=True, seed=seed)
    forest.fit(split.X_train, split.y_train)
    owner = _owner(forest)
    p = len(owner)
    m = max(1, int(modeling.FOREST_MAX_FEATURES * p))

    bag_cuts = _first_cuts(bagged, owner, FIRST_SPLIT_SAMPLE)
    forest_cuts = _first_cuts(forest, owner, FIRST_SPLIT_SAMPLE)
    bag_rho, bag_s2 = _tree_correlation(
        split, lambda n, s: modeling.bagging_pipeline(n_estimators=n, seed=s), seed=seed
    )
    rf_rho, rf_s2 = _tree_correlation(
        split, lambda n, s: modeling.forest_pipeline(n_estimators=n, seed=s), seed=seed
    )

    sweep = sorted({1, 2, 4, 8, m, p // 2, p})
    curve = pd.DataFrame(
        {
            "m": sweep,
            "test_mae": [
                modeling.score(
                    modeling.forest_pipeline(
                        n_estimators=sweep_trees, max_features=int(k), seed=seed
                    ).fit(split.X_train, split.y_train),
                    split.X_test,
                    split.y_test,
                )[0]
                for k in sweep
            ],
        }
    )
    train_mae = modeling.mae(split.y_train, forest.predict(split.X_train))
    test_mae, test_r2 = modeling.score(forest, split.X_test, split.y_test)
    path = _write(curve, args, "part4_mtry.csv")
    if args.charts:
        _plot_lines(
            curve,
            "m",
            {"test MAE": "test_mae"},
            args.output_dir / "part4_mtry.png",
            title=f"Part 4 — random forest: error vs columns per split (of {p})",
            xlabel="m (candidate columns per split)",
        )

    def _top(cuts: Counter[str]) -> str:
        return ", ".join(f"{c} {n}" for c, n in cuts.most_common(4))

    print(f"[Part 4] first cut of {FIRST_SPLIT_SAMPLE} bagged trees: {_top(bag_cuts)}")
    print(f"[Part 4] first cut of {FIRST_SPLIT_SAMPLE} forest trees (m={m}/{p}): ", end="")
    print(_top(forest_cuts))
    for label, rho, s2 in (("bagging", bag_rho, bag_s2), ("forest ", rf_rho, rf_s2)):
        floor = rho * s2 + (1 - rho) * s2 / n_estimators
        print(
            f"[Part 4]   {label} tree correlation rho={rho:.2f}  one tree's variance "
            f"sigma2={s2:,.0f}  -> average of {n_estimators}: {floor:,.0f} "
            f"(sd {floor**0.5:.1f} EUR)"
        )
    print(f"[Part 4] forest: train MAE={train_mae:.2f}  test MAE={test_mae:.2f}")
    print(f"[Part 4]   wrote {path}" + (" (+ PNG)" if args.charts else ""))
    return (
        modeling.PartResult(
            4,
            f"random forest (m={m} of {p})",
            train_mae,
            test_mae,
            test_r2,
            note=f"tree correlation {bag_rho:.2f} -> {rf_rho:.2f}",
        ),
        forest,
    )


# --- Part 5 — the free test ------------------------------------------------
def run_part5(split: modeling.Split, forest: Pipeline) -> modeling.PartResult:
    """Out-of-bag error: score each training listing with only the trees that never saw it.

    About 37% of the trees left any given listing out of their bootstrap sample, so they can
    grade it as if it were new. The resulting error matches the held-out test error without
    setting a single listing aside.
    """
    Xd = modeling.design(forest, split.X_train)
    oob_pred, n_trees = modeling.out_of_bag_predict(forest, Xd)
    graded = n_trees > 0
    oob_mae = modeling.mae(split.y_train[graded], oob_pred[graded])
    reference = forest.named_steps["est"].oob_prediction_
    agreement = float(np.nanmax(np.abs(oob_pred[graded] - reference[graded])))
    test_mae, test_r2 = modeling.score(forest, split.X_test, split.y_test)
    b = len(forest.named_steps["est"].estimators_)

    print(
        f"[Part 5] listing #0 was left out by {int(n_trees[0])} of {b} trees; "
        f"on average {float(n_trees.mean()) / b:.1%} (e^-1 = {np.exp(-1):.1%})"
    )
    print(
        f"[Part 5] OOB MAE={oob_mae:.2f}  vs  held-out test MAE={test_mae:.2f}  "
        f"(gap {oob_mae - test_mae:+.2f} EUR)"
    )
    print(f"[Part 5]   matches sklearn's oob_prediction_ to {agreement:.1e}")
    return modeling.PartResult(
        5,
        "random forest, OOB check",
        None,
        test_mae,
        test_r2,
        note=f"OOB MAE {oob_mae:.2f}, no test set needed",
    )


# --- Part 6 — which columns really matter -------------------------------------
def run_part6(
    split: modeling.Split,
    forest: Pipeline,
    args: argparse.Namespace,
    *,
    seed: int = config.SEED,
) -> modeling.PartResult:
    """Permutation importance on the out-of-bag rows, against the split-gain importance.

    Shuffle one column across the listings (all dummies of a categorical together), re-score
    with the out-of-bag trees, and record how many euro of error it adds. Writes
    ``part6_importance.csv``.
    """
    Xd = modeling.design(forest, split.X_train)
    groups = modeling.column_groups(forest)
    base_pred, n_trees = modeling.out_of_bag_predict(forest, Xd)
    graded = n_trees > 0
    base = modeling.mae(split.y_train[graded], base_pred[graded])
    gain = forest.named_steps["est"].feature_importances_
    rng = np.random.default_rng(seed)

    rows = []
    for column, idx in groups.items():
        shuffled = Xd.copy()
        shuffled[:, idx] = Xd[rng.permutation(len(Xd))][:, idx]
        pred, _ = modeling.out_of_bag_predict(forest, shuffled)
        rows.append(
            {
                "column": column,
                "mae_increase": modeling.mae(split.y_train[graded], pred[graded]) - base,
                "split_gain_share": float(gain[idx].sum()),
            }
        )
    table = pd.DataFrame(rows).sort_values("mae_increase", ascending=False, ignore_index=True)
    path = _write(table, args, "part6_importance.csv")
    if args.charts:
        _plot_bars(
            table,
            "column",
            "mae_increase",
            args.output_dir / "part6_importance.png",
            title="Part 6 — extra error when the column is shuffled (OOB)",
            xlabel="MAE increase (EUR)",
        )

    by_gain = table.sort_values("split_gain_share", ascending=False)["column"].tolist()
    print(f"[Part 6] OOB MAE={base:.2f}; shuffling each column adds:")
    for row in table.itertuples(index=False):
        print(
            f"[Part 6]   {row.column:<22} +{row.mae_increase:6.2f} EUR   "
            f"split-gain share {row.split_gain_share:6.1%}"
        )
    print(f"[Part 6] split-gain ranking: {', '.join(by_gain[:5])}")
    print(f"[Part 6]   wrote {path}" + (" (+ PNG)" if args.charts else ""))
    head = table.iloc[0]
    tail = table.iloc[-1]
    return modeling.PartResult(
        6,
        "forest, permutation importance",
        None,
        None,
        None,
        note=f"{head['column']} +{head['mae_increase']:.0f} EUR ... "
        f"{tail['column']} +{tail['mae_increase']:.1f}",
    )


# --- Part 7 — the tree that learns from the previous one's mistakes -------------
def _tune_rounds(
    split: modeling.Split, loss: str, max_iter: int, seed: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Pick the number of boosting rounds on a validation slice of the training rows.

    Returns ``(best_rounds, validation_curve, test_curve)``; the test curve is traced for the
    chart only and plays no part in the choice.
    """
    X_in, X_val, y_in, y_val = train_test_split(
        split.X_train, split.y_train, test_size=0.25, random_state=seed
    )
    model = modeling.boosting_pipeline(loss=loss, max_iter=max_iter, seed=seed)
    model.fit(X_in, np.asarray(y_in))
    booster = model.named_steps["est"]
    val = np.array(
        [
            modeling.mae(np.asarray(y_val), p)
            for p in booster.staged_predict(modeling.design(model, X_val))
        ]
    )
    test = np.array(
        [
            modeling.mae(split.y_test, p)
            for p in booster.staged_predict(modeling.design(model, split.X_test))
        ]
    )
    return int(np.argmin(val)) + 1, val, test


def run_part7(
    split: modeling.Split,
    args: argparse.Namespace,
    *,
    max_iter: int = BOOSTING_MAX_ITER,
    seed: int = config.SEED,
) -> list[modeling.PartResult]:
    """Gradient boosting: short trees added one by one, each fitted to the remaining error.

    ``f(x) = sum_m nu * T_m(x)``. With squared loss each tree fits the residuals of the sum so
    far; more rounds lower the bias until the trees start fitting noise, so the number of rounds
    is a complexity dial, chosen here on validation rows. Then the same machine with absolute
    loss, which aims at the euro error directly. Writes ``part7_boosting.csv``.
    """
    results: list[modeling.PartResult] = []
    curves: dict[str, np.ndarray] = {"rounds": np.arange(1, max_iter + 1)}
    for loss, label in (("squared_error", "squared"), ("absolute_error", "absolute")):
        rounds, val, test = _tune_rounds(split, loss, max_iter, seed)
        curves[f"val_mae_{label}"] = val
        curves[f"test_mae_{label}"] = test
        model = modeling.boosting_pipeline(loss=loss, max_iter=rounds, seed=seed)
        model.fit(split.X_train, split.y_train)
        train_mae = modeling.mae(split.y_train, model.predict(split.X_train))
        test_mae, test_r2 = modeling.score(model, split.X_test, split.y_test)
        print(
            f"[Part 7] {label} loss, validation MAE by round: 1 -> {val[0]:.1f}, "
            f"10 -> {val[9]:.1f}, 100 -> {val[99]:.1f}, {rounds} -> {val[rounds - 1]:.2f} (best), "
            f"{max_iter} -> {val[-1]:.2f}"
        )
        print(
            f"[Part 7]   refit with {rounds} rounds: train MAE={train_mae:.2f}  "
            f"test MAE={test_mae:.2f}"
        )
        results.append(
            modeling.PartResult(
                7,
                f"boosting, {label} loss (M={rounds})",
                train_mae,
                test_mae,
                test_r2,
                note=f"{max_iter} rounds: val {val[-1]:.1f} (overfits)"
                if val[-1] > val[rounds - 1] + 0.5
                else "",
            )
        )
    curve = pd.DataFrame(curves)
    path = _write(curve, args, "part7_boosting.csv")
    if args.charts:
        _plot_lines(
            curve,
            "rounds",
            {k: k for k in curve.columns if k != "rounds"},
            args.output_dir / "part7_boosting.png",
            title="Part 7 — boosting: error vs number of trees",
            xlabel="rounds (trees)",
            logx=True,
        )
    print(f"[Part 7]   wrote {path}" + (" (+ PNG)" if args.charts else ""))
    return results


# --- Part 8 — no method always wins -------------------------------------------
def lineup(
    numeric: list[str], categorical: list[str], *, seed: int = config.SEED
) -> dict[str, Pipeline]:
    """Every method of both series, each with one fixed recipe applied to any dataset."""
    return {
        "mean": modeling.mean_pipeline(numeric, categorical),
        "kNN (k=10)": modeling.knn_pipeline(numeric, categorical, k=10),
        "OLS": modeling.ols_pipeline(numeric, categorical),
        "Lasso": modeling.lasso_pipeline(numeric, categorical, seed=seed),
        "tree": modeling.tree_pipeline(numeric, categorical, min_samples_leaf=20, seed=seed),
        "bagging": modeling.bagging_pipeline(numeric, categorical, n_estimators=100, seed=seed),
        "random forest": modeling.forest_pipeline(numeric, categorical, seed=seed),
        # Early stopping lets one recipe size itself to 442 rows or to 30,000.
        "boosting": modeling.boosting_pipeline(
            numeric,
            categorical,
            loss="absolute_error",
            max_iter=BOOSTING_MAX_ITER,
            early_stopping=True,
            seed=seed,
        ),
    }


def _leaderboard(
    models: dict[str, Pipeline], X: pd.DataFrame, y: np.ndarray, seed: int
) -> pd.DataFrame:
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=folds, scoring="neg_mean_absolute_error")
        rows.append({"model": name, "cv_mae": float(-scores.mean())})
    board = pd.DataFrame(rows).sort_values("cv_mae", ignore_index=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


def run_part8(
    listings: pd.DataFrame, args: argparse.Namespace, *, seed: int = config.SEED
) -> modeling.PartResult:
    """The same lineup, ranked by 5-fold CV on Rome and on a dataset with another structure.

    The second dataset is scikit-learn's bundled *diabetes* study (442 patients, 10 measured
    variables, disease progression one year later), where the signal is close to additive and
    linear. The ranking flips: what wins depends on whether a method's assumptions match the
    data, not on how sophisticated it is. Writes ``part8_nfl.csv``.
    """
    X, y = modeling.make_xy(listings)
    rome = _leaderboard(lineup(modeling.NUMERIC, modeling.CATEGORICAL, seed=seed), X, y, seed)
    diabetes = load_diabetes(as_frame=True)
    Xd = diabetes.data
    yd = diabetes.target.to_numpy()
    other = _leaderboard(lineup(list(Xd.columns), [], seed=seed), Xd, yd, seed)
    board = pd.concat(
        [rome.assign(dataset="airbnb_rome"), other.assign(dataset="diabetes")], ignore_index=True
    )
    path = _write(board[["dataset", "rank", "model", "cv_mae"]], args, "part8_nfl.csv")

    def _line(frame: pd.DataFrame) -> str:
        return " > ".join(f"{r.model} {r.cv_mae:.1f}" for r in frame.itertuples(index=False))

    def _rank_of(frame: pd.DataFrame, model: str) -> int:
        return int(frame.loc[frame["model"] == model, "rank"].iloc[0])

    print(f"[Part 8] Rome (EUR/night):      {_line(rome)}")
    print(f"[Part 8] diabetes (score pts):  {_line(other)}")
    print(f"[Part 8]   wrote {path}")
    rome_best, other_best = rome.iloc[0], other.iloc[0]
    return modeling.PartResult(
        8,
        f"no free lunch ({CV_FOLDS}-fold CV)",
        None,
        float(rome_best["cv_mae"]),
        None,
        note=f"Rome #1 {rome_best['model']} (OLS #{_rank_of(rome, 'OLS')}); "
        f"diabetes #1 {other_best['model']} "
        f"({rome_best['model']} #{_rank_of(other, str(rome_best['model']))})",
    )


# --- Charts -----------------------------------------------------------------
def _plot_lines(
    frame: pd.DataFrame,
    x: str,
    series: dict[str, str],
    path: Path,
    *,
    title: str,
    xlabel: str,
    logx: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, column in series.items():
        ax.plot(frame[x], frame[column], label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE (EUR/night)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_bars(
    frame: pd.DataFrame, label: str, value: str, path: Path, *, title: str, xlabel: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = frame.sort_values(value)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ordered[label], ordered[value])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- Orchestrator -----------------------------------------------------------
def main(argv: list[str] | None = None) -> list[modeling.PartResult]:
    """Load once, carve one fixed split, run Parts 1->8 and print the summary table."""
    args = parse_cli(argv)
    print(f"Loading {args.listings} ...")
    listings = modeling.load_listings(args.listings)
    print(
        f"Loaded {len(listings):,} listings; carving the one fixed {1 - args.test_size:.0%}/"
        f"{args.test_size:.0%} split (seed={args.seed})."
    )
    split = modeling.make_split(listings, test_size=args.test_size, seed=args.seed)

    results: list[modeling.PartResult] = []
    results.extend(run_part1(split, seed=args.seed))
    results.append(run_part2(split, args, seed=args.seed))
    part3, bagged = run_part3(split, args, seed=args.seed)
    results.append(part3)
    part4, forest = run_part4(split, args, bagged, seed=args.seed)
    results.append(part4)
    results.append(run_part5(split, forest))
    results.append(run_part6(split, forest, args, seed=args.seed))
    results.extend(run_part7(split, args, seed=args.seed))
    results.append(run_part8(listings, args, seed=args.seed))

    print()
    print(modeling.format_report(results))
    return results


if __name__ == "__main__":
    main()
