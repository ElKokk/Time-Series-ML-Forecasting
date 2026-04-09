"""Contract test for the _predict_week helper extracted in the cleanup pass.

The recursive forecast loop calls _predict_week once per forecast week.
Boundary weeks can hand the helper a feature matrix that is missing some
of the interaction columns the booster was trained on (because those
columns depend on lag features that are NaN at the very edge of the
history). The helper is responsible for zero-filling those missing
columns and reordering the rest to the booster's expected order before
calling model.predict.

This test pins both behaviours so a future refactor of the helper cannot
silently shift the column alignment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from forecast import _predict_week


def _train_toy_model(seed: int = 0) -> XGBRegressor:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feature_a": rng.normal(0, 1, 80),
            "feature_b": rng.normal(0, 1, 80),
            "feature_c": rng.normal(0, 1, 80),
        }
    )
    y = 2.0 * X["feature_a"] - 1.0 * X["feature_b"] + 0.5 * X["feature_c"]
    model = XGBRegressor(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


def test_predict_week_zero_fills_missing_columns_and_reorders():
    model = _train_toy_model()
    booster_features = list(model.get_booster().feature_names)

    # Test set is missing feature_c entirely AND has the remaining columns
    # in the wrong order. Both are situations the helper must repair.
    X_test = pd.DataFrame(
        {
            "feature_b": [0.10, 0.20, -0.30],
            "feature_a": [0.50, -0.30, 0.80],
        }
    )

    predictions = _predict_week(model, X_test, booster_features)

    assert predictions.shape == (3,)
    assert not np.any(np.isnan(predictions))

    # The expected behaviour is "missing columns become 0, then reorder to
    # the booster's column list". So predicting on an explicitly-zero-filled
    # version with the columns in booster order must give the same answer.
    X_filled = X_test.copy()
    X_filled["feature_c"] = 0.0
    X_filled = X_filled[booster_features].astype(float)
    direct = model.predict(X_filled.values)

    np.testing.assert_array_almost_equal(predictions, direct)


def test_predict_week_handles_single_row_input():
    """The recursive loop predicts day-by-day, so 1-row inputs must work."""
    model = _train_toy_model()
    booster_features = list(model.get_booster().feature_names)

    X_test = pd.DataFrame(
        {"feature_a": [0.42], "feature_b": [-0.17], "feature_c": [0.05]}
    )
    predictions = _predict_week(model, X_test, booster_features)

    assert predictions.shape == (1,)
    assert np.isfinite(predictions[0])
