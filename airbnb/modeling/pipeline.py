"""Shared helpers for the seven-part tree-ensemble arc (README "The model, part by part").

Single source of truth for loading the snapshot, the one fixed train/test split, the
scikit-learn preprocessing and model pipelines, the metrics, the per-tree plumbing the ensemble
parts need (tree-by-tree predictions, out-of-bag predictions, raw-column groups) and the final
summary table. Every ``run_partN`` in :mod:`airbnb.modeling.arc` builds its models here, so the
preprocessing is identical across parts.

The metric is the mean absolute error in euro: "on average the suggested price is off by X EUR
a night" is the sentence a host understands.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from airbnb import paths
from airbnb.domain import config

SEED = config.SEED
TARGET = config.TARGET
NUMERIC = config.NUMERIC_FEATURES
CATEGORICAL = config.CATEGORICAL_FEATURES
FEATURES = config.FEATURES

N_BAGGED_TREES = 200
N_FOREST_TREES = 300
# ESL's default for regression forests: a third of the predictors are candidates at each split.
FOREST_MAX_FEATURES = 1 / 3
BOOSTING_LEARNING_RATE = 0.1
# Short trees, as ESL recommends for boosting (4 <= J <= 8): each one only nudges the sum.
BOOSTING_LEAVES = 8
# Early stopping holds back a tenth of the rows and halts once this many rounds add no gain.
BOOSTING_EARLY_STOPPING_ROUNDS = 20
BOOSTING_VALIDATION_FRACTION = 0.1

_DTYPES: dict[str, Any] = {
    **{c: "float64" for c in NUMERIC},
    **{c: str for c in CATEGORICAL},
    TARGET: "float64",
}


class Split(NamedTuple):
    """One train/test partition; supports both ``a, b, c, d = split`` and ``split.X_train``."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass
class PartResult:
    """One row of the final summary table (``None`` where a metric does not apply)."""

    part: int
    name: str
    mae_train: float | None
    mae_test: float | None
    r2_test: float | None
    note: str = ""


# --- Data loading & splitting ----------------------------------------------
def load_listings(path: Path | str = paths.LISTINGS_CSV) -> pd.DataFrame:
    """Read the cleaned snapshot, keeping the features and the target in contract order."""
    frame = pd.read_csv(path, usecols=FEATURES + [TARGET], dtype=_DTYPES)
    return cast(pd.DataFrame, frame[FEATURES + [TARGET]])


def file_sha256(path: Path | str) -> str:
    """Return the SHA-256 of a file's bytes: the identity of a snapshot or of a run's output."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_xy(listings: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the feature frame ``X`` and the nightly-price target ``y``."""
    return cast(pd.DataFrame, listings[FEATURES]), listings[TARGET].to_numpy(dtype=float)


def make_split(listings: pd.DataFrame, *, test_size: float = 0.2, seed: int = SEED) -> Split:
    """Carve the one fixed train/test partition Parts 1-7 share.

    The test rows are set aside here and only ever *scored*: every choice a part makes
    (pruning strength, number of boosting rounds) is made on the training rows alone.
    """
    X, y = make_xy(listings)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return Split(
        cast(pd.DataFrame, X_train),
        cast(pd.DataFrame, X_test),
        np.asarray(y_train),
        np.asarray(y_test),
    )


# --- Preprocessing & pipelines ---------------------------------------------
def build_preprocessor(
    numeric: Sequence[str] = NUMERIC,
    categorical: Sequence[str] = CATEGORICAL,
) -> ColumnTransformer:
    """Median-impute the numeric columns and one-hot the categorical ones.

    The tree ensembles are indifferent to scale, so the numeric columns are only imputed.
    Imputation sits inside the pipeline, fitted on the training rows only, and is also what
    fills the columns a host leaves blank in the simulator.
    """
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", Pipeline(numeric_steps), list(numeric))
    ]
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        transformers.append(("cat", encoder, list(categorical)))
    return ColumnTransformer(transformers)


def make_pipeline(
    estimator: RegressorMixin,
    numeric: Sequence[str] = NUMERIC,
    categorical: Sequence[str] = CATEGORICAL,
) -> Pipeline:
    """``Pipeline([("pre", preprocessor), ("est", estimator)])``.

    The step names are load-bearing: the ensemble helpers reach the fitted trees through
    ``"est"`` and the transformed design through ``"pre"``.
    """
    pre = build_preprocessor(numeric, categorical)
    return Pipeline([("pre", pre), ("est", estimator)])


def mean_pipeline(numeric: Sequence[str] = NUMERIC, categorical: Sequence[str] = CATEGORICAL):
    """The no-model baseline: always answer the training mean."""
    return make_pipeline(DummyRegressor(strategy="mean"), numeric, categorical)


def tree_pipeline(
    numeric: Sequence[str] = NUMERIC,
    categorical: Sequence[str] = CATEGORICAL,
    *,
    seed: int = SEED,
    **tree_params: Any,
) -> Pipeline:
    """A CART regression tree (squared-error splits); ``tree_params`` go to the tree."""
    tree = DecisionTreeRegressor(random_state=seed, **tree_params)
    return make_pipeline(tree, numeric, categorical)


def bagging_pipeline(
    numeric: Sequence[str] = NUMERIC,
    categorical: Sequence[str] = CATEGORICAL,
    *,
    n_estimators: int = N_BAGGED_TREES,
    seed: int = SEED,
) -> Pipeline:
    """Bagged full-depth trees: each grown on its own bootstrap sample, predictions averaged."""
    bagging = BaggingRegressor(
        DecisionTreeRegressor(), n_estimators=n_estimators, n_jobs=-1, random_state=seed
    )
    return make_pipeline(bagging, numeric, categorical)


def forest_pipeline(
    numeric: Sequence[str] = NUMERIC,
    categorical: Sequence[str] = CATEGORICAL,
    *,
    n_estimators: int = N_FOREST_TREES,
    max_features: float | int = FOREST_MAX_FEATURES,
    oob_score: bool = False,
    seed: int = SEED,
) -> Pipeline:
    """A random forest: bagging plus a random subset of ``max_features`` columns per split."""
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        oob_score=oob_score,
        n_jobs=-1,
        random_state=seed,
    )
    return make_pipeline(forest, numeric, categorical)


class _EarlyStoppingXGB(XGBRegressor):
    """An ``XGBRegressor`` that carves its own validation slice so it can stop early.

    A scikit-learn pipeline hands the final estimator the already-transformed design, so the
    rows XGBoost needs to watch for early stopping are split off here rather than passed in by
    the caller. This keeps ``boosting_pipeline(...).fit(X, y)`` the same two-argument call as
    every other model in the arc.
    """

    def fit(self, X: Any, y: Any) -> _EarlyStoppingXGB:
        """Fit on a random nine tenths of the rows, watching the rest to decide when to stop.

        Deliberately takes only ``X`` and ``y``: the arc never routes per-sample fit params
        (``sample_weight`` and friends) through the pipeline, and accepting them here would
        hand XGBoost arrays the internal split has already shortened.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=float)
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=BOOSTING_VALIDATION_FRACTION, random_state=self.random_state
        )
        super().fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
        return self


def boosting_estimator(
    *,
    loss: str = "squared_error",
    quantile: float | None = None,
    max_iter: int = 1000,
    learning_rate: float = BOOSTING_LEARNING_RATE,
    max_leaf_nodes: int = BOOSTING_LEAVES,
    eval_metric: str | None = None,
    early_stopping: bool = False,
    seed: int = SEED,
) -> XGBRegressor:
    """Build the unfitted XGBoost regressor the arc's boosting part and the simulator share.

    With ``loss="squared_error"`` each tree fits the plain residuals; ``"absolute_error"`` fits
    their sign, which is what minimizes the euro error a host sees; a ``quantile`` predicts the
    price that a fraction ``q`` of similar listings stay under. ``max_leaf_nodes`` short trees
    are grown leaf-wise (``grow_policy="lossguide"``). With ``early_stopping`` the estimator
    sizes itself on a held-out slice (see :class:`_EarlyStoppingXGB`); without it the caller
    fixes ``max_iter`` rounds, which is how the arc traces the error round by round. ``eval_metric``
    overrides the metric XGBoost watches (the arc reads it off in euro as ``"mae"``).
    """
    objectives = {"squared_error": "reg:squarederror", "absolute_error": "reg:absoluteerror"}
    params: dict[str, Any] = dict(
        n_estimators=max_iter,
        learning_rate=learning_rate,
        max_leaves=max_leaf_nodes,
        max_depth=0,
        grow_policy="lossguide",
        random_state=seed,
        n_jobs=-1,
    )
    if eval_metric is not None:
        params["eval_metric"] = eval_metric
    if quantile is not None:
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = quantile
    elif loss in objectives:
        params["objective"] = objectives[loss]
    else:
        raise ValueError(
            f"unknown loss {loss!r}; use 'squared_error' or 'absolute_error', "
            "or pass quantile=<float> for a quantile model"
        )
    if early_stopping:
        return _EarlyStoppingXGB(early_stopping_rounds=BOOSTING_EARLY_STOPPING_ROUNDS, **params)
    return XGBRegressor(**params)


def boosting_pipeline(
    numeric: Sequence[str] = NUMERIC,
    categorical: Sequence[str] = CATEGORICAL,
    *,
    loss: str = "squared_error",
    quantile: float | None = None,
    max_iter: int = 1000,
    learning_rate: float = BOOSTING_LEARNING_RATE,
    max_leaf_nodes: int = BOOSTING_LEAVES,
    early_stopping: bool = False,
    seed: int = SEED,
) -> Pipeline:
    """Preprocess, then gradient-boost (see :func:`boosting_estimator` for the loss and sizing).

    ``eval_metric`` is not exposed here: it only matters when reading a booster's
    ``evals_result_`` round by round, which the arc does on the raw estimator (see
    :func:`boosting_eval_curves`), never through a fitted pipeline.
    """
    booster = boosting_estimator(
        loss=loss,
        quantile=quantile,
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        early_stopping=early_stopping,
        seed=seed,
    )
    return make_pipeline(booster, numeric, categorical)


def boosting_eval_curves(
    Xd_fit: np.ndarray,
    y_fit: np.ndarray,
    Xd_val: np.ndarray,
    y_val: np.ndarray,
    Xd_test: np.ndarray,
    y_test: np.ndarray,
    *,
    loss: str,
    max_iter: int,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a booster for ``max_iter`` rounds and return its per-round MAE on the val and test rows.

    The val and test rows are only *scored* each round — there is no early stopping, so neither
    steers the fit: the arc reads the val curve to pick the number of rounds and plots the test
    curve to show the overfitting. The booster is the round-tuning model, fitted on ``Xd_fit``
    alone, so its test curve is that model's own trajectory, not the same quantity as the final
    refit's test MAE. The ``evals_result_`` keys live here so the orchestrator stays free of
    XGBoost internals.

    Args:
        Xd_fit: the inner-training design (already preprocessed) the booster learns from.
        y_fit: its targets.
        Xd_val: the validation design whose curve chooses the number of rounds.
        y_val: its targets.
        Xd_test: the held-out test design, scored each round for the chart only.
        y_test: its targets.
        loss: ``"squared_error"`` or ``"absolute_error"``.
        max_iter: how many rounds to grow and trace.
        seed: the booster's ``random_state``.

    Returns:
        ``(val_mae, test_mae)``, each a ``max_iter``-long array of the MAE after every round.
    """
    booster = boosting_estimator(loss=loss, max_iter=max_iter, eval_metric="mae", seed=seed)
    booster.fit(Xd_fit, y_fit, eval_set=[(Xd_val, y_val), (Xd_test, y_test)], verbose=False)
    history = booster.evals_result_
    return (
        np.asarray(history["validation_0"]["mae"], dtype=float),
        np.asarray(history["validation_1"]["mae"], dtype=float),
    )


# --- Metrics ----------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the mean absolute error (EUR per night)."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the coefficient of determination R²."""
    return float(r2_score(y_true, y_pred))


def score(model: Any, X: pd.DataFrame, y: np.ndarray) -> tuple[float, float]:
    """Return ``(mae, r2)`` of a fitted ``model`` on ``(X, y)``."""
    y_pred = np.asarray(model.predict(X))
    return mae(y, y_pred), r2(y, y_pred)


# --- Ensemble plumbing --------------------------------------------------------
def design(fitted: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return ``X`` as the dense float matrix the fitted pipeline's estimator sees."""
    return np.asarray(fitted.named_steps["pre"].transform(X), dtype=np.float32)


def column_groups(fitted: Pipeline) -> dict[str, list[int]]:
    """Map each raw feature to the design columns it became (one-hot categories expand).

    Permuting or crediting a categorical feature means touching all of its dummy columns at
    once, which is what this mapping is for.
    """
    pre = fitted.named_steps["pre"]
    names = [str(n) for n in pre.get_feature_names_out()]
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        branch, column = name.split("__", 1)
        if branch == "cat":
            column = next(c for c in pre.transformers_[1][2] if column.startswith(f"{c}_"))
        groups.setdefault(column, []).append(i)
    return groups


def column_owner(fitted: Pipeline) -> dict[int, str]:
    """Map each design column index back to the raw feature it came from."""
    return {i: col for col, idx in column_groups(fitted).items() for i in idx}


def design_names(fitted: Pipeline) -> list[str]:
    """Return the design columns' readable names (``room_type=Private room``, ``bedrooms``)."""
    out: list[str] = []
    pre = fitted.named_steps["pre"]
    raw_names = [str(n).split("__", 1)[1] for n in pre.get_feature_names_out()]
    owner = column_owner(fitted)
    for i, name in enumerate(raw_names):
        column = owner[i]
        out.append(column if name == column else f"{column}={name[len(column) + 1 :]}")
    return out


def tree_predictions(fitted: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return a ``(n_trees, n_rows)`` matrix: each tree of a fitted bagging/forest, row by row."""
    ensemble = fitted.named_steps["est"]
    Xd = design(fitted, X)
    feature_sets = getattr(ensemble, "estimators_features_", None)
    preds = []
    for i, tree in enumerate(ensemble.estimators_):
        cols = feature_sets[i] if feature_sets is not None else slice(None)
        preds.append(tree.predict(Xd[:, cols]))
    return np.vstack(preds)


def out_of_bag_predict(fitted: Pipeline, Xd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average, for every training row, only the trees whose bootstrap sample left it out.

    Args:
        fitted: a bagging or forest pipeline fitted on exactly the rows of ``Xd``.
        Xd: the training design matrix (``design(fitted, X_train)``), possibly with some
            columns permuted.

    Returns:
        ``(prediction, n_trees)`` per row. A row every tree happened to draw has
        ``n_trees == 0`` and a ``NaN`` prediction.
    """
    ensemble = fitted.named_steps["est"]
    n = Xd.shape[0]
    total = np.zeros(n)
    count = np.zeros(n)
    for tree, in_bag in zip(ensemble.estimators_, ensemble.estimators_samples_):
        out = np.ones(n, dtype=bool)
        out[in_bag] = False
        total[out] += tree.predict(Xd[out])
        count[out] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return total / count, count


# --- Reporting --------------------------------------------------------------
def format_report(results: list[PartResult]) -> str:
    """Return the fixed-width summary ``Part | model | train MAE | test MAE | test R² | note``.

    Part 0 is the no-model baseline the header quotes, so every later row reads as "how much
    of the baseline error this part removed".
    """
    baseline = next((r.mae_test for r in results if r.part == 0), None)
    lines: list[str] = []
    if baseline is not None:
        lines.append(
            f"Baseline (always answer the mean) test MAE = {baseline:.2f} EUR/night; "
            "MAE in EUR, R^2 on the held-out test set."
        )
    else:
        lines.append("MAE in EUR/night, R^2 on the held-out test set.")
    head = (
        f"{'Part':>4} | {'model':<34} | {'train MAE':>9} | {'test MAE':>8} | {'test R2':>7} | note"
    )
    lines.append(head)
    lines.append("-" * len(head))
    for r in sorted(results, key=lambda x: x.part):
        tr = "-" if r.mae_train is None else f"{r.mae_train:.2f}"
        te = "-" if r.mae_test is None else f"{r.mae_test:.2f}"
        r2v = "-" if r.r2_test is None else f"{r.r2_test:.3f}"
        lines.append(f"{r.part:>4} | {r.name:<34} | {tr:>9} | {te:>8} | {r2v:>7} | {r.note}")
    return "\n".join(lines)
