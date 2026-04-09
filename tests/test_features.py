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

    A lag of 5 or less would reach into another row inside the same forecast
    week, which is self-reference: the row at offset 5 would be reading a
    value the model is also trying to predict in the same iteration.

    Lags between 6 and the full lead time (e.g. 6, 7, 8 in this pipeline)
    are deliberately allowed and are NOT leakage, even though they reach
    into days that lie inside the 2-week lead window. The recursive forecast
    loop folds each week's predictions back into the history before training
    the next week, so by the time the model is forecasting week N every
    value that lag_6 reaches into is either a real actual (for week 1) or
    the model's own previously-emitted prediction (for weeks 2 and 3) -
    never a real future value the model "shouldn't see".
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
