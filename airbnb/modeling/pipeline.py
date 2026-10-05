"""Shared helpers for the eight-part tree-ensemble arc (README "The model, part by part").

Single source of truth for loading the snapshot, the one fixed train/test split, the
scikit-learn preprocessing and model pipelines, the metrics, the per-tree plumbing the ensemble
parts need (tree-by-tree predictions, out-of-bag predictions, raw-column groups) and the final
summary table. Every ``run_partN`` in :mod:`airbnb.modeling.arc` builds its models here, so the
preprocessing is identical across parts.

The metric is the mean absolute error in euro: "on average the suggested price is off by X EUR
a night" is the sentence a host understands.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import BaggingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from airbnb.domain import config
from airbnb.infrastructure import paths

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
    *,
    scale: bool = False,
) -> ColumnTransformer:
    """Median-impute the numeric columns and one-hot the categorical ones.

    Trees are indifferent to scale, so ``scale`` is only for the distance- and
    coefficient-based models of Part 8. Imputation sits inside the pipeline, fitted on the
    training rows only, and is also what fills the columns a host leaves blank in the simulator.
    """
    numeric_steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
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
    *,
    scale: bool = False,
) -> Pipeline:
    """``Pipeline([("pre", preprocessor), ("est", estimator)])``.

    The step names are load-bearing: the ensemble helpers reach the fitted trees through
    ``"est"`` and the transformed design through ``"pre"``.
    """
    pre = build_preprocessor(numeric, categorical, scale=scale)
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
    """Gradient boosting: ``max_iter`` short trees, each fitted to what the sum still gets wrong.

    With ``loss="squared_error"`` each tree fits the plain residuals; ``"absolute_error"`` fits
    their sign, which is what minimizes the euro error a host sees; ``"quantile"`` with
    ``quantile=q`` predicts the price that a fraction ``q`` of similar listings stay under.
    Early stopping is off by default so the arc can trace the error round by round.
    """
    params: dict[str, Any] = {}
    if quantile is not None:
        params["quantile"] = quantile
    booster = HistGradientBoostingRegressor(
        loss=cast(Any, loss),
        max_iter=max_iter,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        early_stopping=early_stopping,
        random_state=seed,
        **params,
    )
    return make_pipeline(booster, numeric, categorical)


def knn_pipeline(
    numeric: Sequence[str] = NUMERIC,
    categorical: Sequence[str] = CATEGORICAL,
    *,
    k: int = 10,
) -> Pipeline:
    """Standardized k-nearest-neighbours regression."""
    return make_pipeline(KNeighborsRegressor(n_neighbors=k), numeric, categorical, scale=True)


def ols_pipeline(numeric: Sequence[str] = NUMERIC, categorical: Sequence[str] = CATEGORICAL):
    """Ordinary least squares on the standardized design."""
    return make_pipeline(LinearRegression(), numeric, categorical, scale=True)


def lasso_pipeline(
    numeric: Sequence[str] = NUMERIC,
    categorical: Sequence[str] = CATEGORICAL,
    *,
    seed: int = SEED,
) -> Pipeline:
    """Lasso with its penalty chosen by 5-fold cross-validation on the training rows."""
    lasso = LassoCV(cv=5, random_state=seed, max_iter=20000)
    return make_pipeline(lasso, numeric, categorical, scale=True)


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
