"""Contracts on the feature engineering pipeline.

These tests pin two properties of ``create_features`` that the recursive
forecast loop silently depends on:

1. Each ``*_lag_k`` column at row ``r`` reflects the source value at row
   ``r - k`` exactly. A bug in shift offsets, fill order, or column
   construction would corrupt the lag math.
2. Every lag offset present in the engineered feature set is at least 6,
   so the model never reaches inside its own forecast week for input.
"""

from __future__ import annotations

import re

from forecast import create_features


_LAG_COL_PATTERN = re.compile(r"^([\w \-]+)_lag_(\d+)$")
_EXPECTED_LAGS = (6, 7, 8, 14, 16, 21)


def test_lag_columns_have_exact_offsets(minimal_dataframe):
    """Every special-lag column must equal source[row - lag] for r >= lag."""
    df = minimal_dataframe.copy()
    out = create_features(df, response_variable="Dry Actuals", product_cols=[])

    n_rows = len(df)
    for lag in _EXPECTED_LAGS:
        col = f"Dry Actuals_lag_{lag}"
        assert col in out.columns, f"missing expected lag column: {col}"

        for row in range(lag, n_rows):
            expected = 1000.0 + (row - lag)
            actual = out[col].iloc[row]
            assert actual == expected, (
                f"{col}[{row}]: expected {expected}, got {actual}"
            )


def test_lag_features_avoid_within_week_leakage(minimal_dataframe):
    """No lag column may have an offset smaller than the forecast week (6 days).

    A lag of 5 or less would let the model see a value from inside the same
    week it is trying to predict, which is target leakage. The forecast
    horizon is 6 business days, so the smallest valid lag is 6.

    NOTE: this test guarantees the *within-week* no-leakage property only.
    The project README describes the model as a 2-week-lead, 1-week-horizon
    forecaster; under the strictest reading of that contract, the smallest
    valid lag would be 14 (or 19 for the last day of the horizon). The
    pipeline does NOT enforce that stricter constraint - lags 6, 7, 8 are
    intentionally present - which means the model is allowed to consume
    data right up to the day before its forecast week begins. Whether that
    matches the operations workflow is a business question; this test only
    pins the architectural property that no row uses a feature derived
    from another row in the same forecast week.
    """
    df = minimal_dataframe.copy()
    out = create_features(df, response_variable="Dry Actuals", product_cols=[])

    lag_offsets = []
    for col in out.columns:
        match = _LAG_COL_PATTERN.match(col)
        if match:
            lag_offsets.append(int(match.group(2)))

    assert lag_offsets, "create_features produced no lag columns at all"
    smallest = min(lag_offsets)
    assert smallest >= 6, (
        f"found a lag column with offset {smallest} < 6 - this would let the "
        f"model see a value from inside its own forecast week"
    )
