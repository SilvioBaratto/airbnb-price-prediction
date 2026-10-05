"""The production predictor and the median fallback, against the ``PricePredictor`` port."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airbnb.application.ports import PricePredictor
from airbnb.domain import config
from airbnb.infrastructure.predictors import MedianPredictor, ModelPredictor
from airbnb.modeling import pipeline
from tests.factories import make_listings


@pytest.fixture(scope="module")
def model() -> ModelPredictor:
    """One fitted production predictor shared by the module (fitting takes a moment)."""
    return ModelPredictor.from_frame(make_listings(600, seed=1))


def test_both_adapters_satisfy_the_port(model: ModelPredictor, listings: pd.DataFrame) -> None:
    """The use case can take either adapter."""
    assert isinstance(model, PricePredictor)
    assert isinstance(MedianPredictor(listings), PricePredictor)


def test_model_predictor_learns_the_price_signal(model: ModelPredictor) -> None:
    """On fresh listings it beats always answering the median by a wide margin."""
    fresh = make_listings(300, seed=2)
    X, y = pipeline.make_xy(fresh)
    model_mae = pipeline.mae(y, model.predict(X))
    median_mae = pipeline.mae(y, np.full_like(y, np.median(y)))
    assert model_mae < 0.6 * median_mae


def test_band_contains_the_suggestion_and_never_goes_below_the_floor(
    model: ModelPredictor, split: pipeline.Split
) -> None:
    """``low <= price <= high`` holds row by row, and nothing dips under the training floor."""
    price = model.predict(split.X_test)
    low, high = model.predict_band(split.X_test)
    assert (low <= price).all() and (price <= high).all()
    assert (low >= config.PRICE_MIN_EUR).all()


def test_predict_ignores_extra_columns_and_column_order(
    model: ModelPredictor, split: pipeline.Split
) -> None:
    """The predictor selects the contract columns by name, so a wider frame is fine."""
    frame = split.X_test.iloc[:5]
    shuffled = frame[list(reversed(frame.columns))].assign(listing_id=1)
    np.testing.assert_allclose(model.predict(shuffled), model.predict(frame))


def test_save_then_load_gives_identical_predictions(
    model: ModelPredictor, split: pipeline.Split, tmp_path: Path
) -> None:
    """Saved weights reload without refitting and answer exactly the same."""
    path = model.save(tmp_path / "nested" / "model.joblib")
    loaded = ModelPredictor.load(path)
    assert loaded.n_train == model.n_train == 600
    np.testing.assert_array_equal(loaded.predict(split.X_test), model.predict(split.X_test))
    for a, b in zip(loaded.predict_band(split.X_test), model.predict_band(split.X_test)):
        np.testing.assert_array_equal(a, b)


def test_median_predictor_answers_each_room_type_median(listings: pd.DataFrame) -> None:
    """Each row gets its room type's median, and an unseen type the overall median."""
    predictor = MedianPredictor(listings)
    frame = pd.DataFrame({"room_type": ["Entire home/apt", "Treehouse"]})
    expected_entire = listings.loc[listings["room_type"] == "Entire home/apt", "price_eur"]
    prices = predictor.predict(frame)
    assert prices[0] == pytest.approx(expected_entire.median())
    assert prices[1] == pytest.approx(listings["price_eur"].median())
    low, high = predictor.predict_band(frame)
    assert (low <= prices).all() and (prices <= high).all()
