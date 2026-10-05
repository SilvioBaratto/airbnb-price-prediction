"""Price predictors — adapters implementing the ``PricePredictor`` port (infrastructure layer).

:class:`ModelPredictor` is the production adapter: the arc's winner, gradient boosting with
absolute loss, for the suggested price, plus two quantile boosters for the band most similar
listings fall in. It can be fitted on construction (retrain-on-launch, a few seconds on the
committed snapshot) or saved once and reloaded instantly with :meth:`save` / :meth:`load`.
:class:`MedianPredictor` is a no-learning fallback and test double: the median price of each
room type.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from airbnb import paths
from airbnb.domain import config
from airbnb.modeling import pipeline

# The band reports where 80% of similar listings sit: wide enough to be honest about how
# scattered real prices are, narrow enough to still guide a host.
BAND_QUANTILES = (0.1, 0.9)
MAX_ROUNDS = 3000


def _booster(loss: str, quantile: float | None = None):
    # Early stopping sizes each booster on its own validation slice, so the three models need
    # no hand-tuned round counts and retraining on a new snapshot stays honest.
    return pipeline.boosting_pipeline(
        loss=loss, quantile=quantile, max_iter=MAX_ROUNDS, early_stopping=True
    )


def _floor(prices: np.ndarray) -> np.ndarray:
    # The model never saw a price below the training slice's floor, so anything lower is an
    # extrapolation artefact, not a price.
    return np.maximum(np.asarray(prices, dtype=float), config.PRICE_MIN_EUR)


class ModelPredictor:
    """A ``PricePredictor`` backed by three gradient-boosting models.

    Args:
        listings_path: the cleaned snapshot to fit on; ignored when ``listings`` is given.
        listings: an in-memory snapshot frame (tests, notebooks).
    """

    def __init__(
        self,
        listings_path: Path | str = paths.LISTINGS_CSV,
        *,
        listings: pd.DataFrame | None = None,
    ) -> None:
        frame = listings if listings is not None else pipeline.load_listings(listings_path)
        X, y = pipeline.make_xy(frame)
        low_q, high_q = BAND_QUANTILES
        self._point = _booster("absolute_error").fit(X, y)
        self._low = _booster("quantile", low_q).fit(X, y)
        self._high = _booster("quantile", high_q).fit(X, y)
        self.n_train = int(len(frame))

    @classmethod
    def from_frame(cls, listings: pd.DataFrame) -> ModelPredictor:
        """Fit directly from an in-memory snapshot frame (no file read)."""
        return cls(listings=listings)

    def save(self, path: Path | str = paths.MODEL_JOBLIB) -> Path:
        """Persist the three fitted models and ``n_train`` with joblib; return the path.

        Creates the parent directory if needed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "point": self._point,
            "low": self._low,
            "high": self._high,
            "n_train": self.n_train,
        }
        joblib.dump(payload, path)
        return path

    @classmethod
    def load(cls, path: Path | str = paths.MODEL_JOBLIB) -> ModelPredictor:
        """Rebuild a predictor from weights written by :meth:`save`, without refitting."""
        payload = joblib.load(Path(path))
        predictor = cls.__new__(cls)
        predictor._point = payload["point"]
        predictor._low = payload["low"]
        predictor._high = payload["high"]
        predictor.n_train = int(payload["n_train"])
        return predictor

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the suggested nightly price (EUR) for each row."""
        return _floor(self._point.predict(frame[config.FEATURES]))

    def predict_band(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(low, high)``, widened where needed so the band contains the suggestion.

        The three boosters are fitted independently, so near the edges of the data a quantile
        model can land on the wrong side of the point model.
        """
        point = self.predict(frame)
        low = np.minimum(_floor(self._low.predict(frame[config.FEATURES])), point)
        high = np.maximum(_floor(self._high.predict(frame[config.FEATURES])), point)
        return low, high


class MedianPredictor:
    """A ``PricePredictor`` that answers each room type's median price, whatever the listing.

    The "no model" reference point, and a cheap, deterministic stand-in for tests. Room types
    absent from the training frame fall back to the overall median.
    """

    def __init__(self, listings: pd.DataFrame) -> None:
        prices = listings.groupby("room_type")[config.TARGET]
        low_q, high_q = BAND_QUANTILES
        self._median = prices.median()
        self._low = prices.quantile(low_q)
        self._high = prices.quantile(high_q)
        overall = listings[config.TARGET]
        self._fallback = (
            float(overall.quantile(low_q)),
            float(overall.median()),
            float(overall.quantile(high_q)),
        )

    def _lookup(self, frame: pd.DataFrame, table: pd.Series, fallback: float) -> np.ndarray:
        return frame["room_type"].map(table).fillna(fallback).to_numpy(dtype=float)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the training median of each row's room type."""
        return self._lookup(frame, self._median, self._fallback[1])

    def predict_band(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return the 10th and 90th percentile of each row's room type."""
        return (
            self._lookup(frame, self._low, self._fallback[0]),
            self._lookup(frame, self._high, self._fallback[2]),
        )
