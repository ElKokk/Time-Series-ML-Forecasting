"""Shared pytest fixtures for the forecast test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _six_day_business_dates(start: str, n: int) -> pd.DatetimeIndex:
    """Return ``n`` consecutive Mon-Sat calendar dates starting from ``start``.

    The forecasting pipeline assumes a 6-day work week (no Sundays), so test
    fixtures use the same calendar to avoid spurious shape mismatches.
    """
    over_sampled = pd.date_range(start=start, periods=n * 2, freq="D")
    return over_sampled[over_sampled.dayofweek < 6][:n]


@pytest.fixture
def minimal_dataframe() -> pd.DataFrame:
    """A 40-row dataframe with the columns ``create_features`` requires.

    The ``Dry Actuals`` column carries a strictly monotonic ramp
    (``1000 + row_index``) so lag values are trivially predictable: the
    value at row ``r`` of ``Dry Actuals_lag_k`` should equal ``1000 + (r-k)``
    for any ``r >= k``.

    40 rows is enough headroom for the largest lag offset (21) plus a buffer
    for the rolling/EMA features that look back 24 days.
    """
    n_rows = 40
    dates = _six_day_business_dates("2024-01-01", n_rows)
    return pd.DataFrame(
        {
            "Date": dates,
            "Day of Week": dates.strftime("%A"),
            "Dry Actuals": np.arange(1000.0, 1000.0 + n_rows),
            "month": dates.month,
            "week": dates.isocalendar().week.astype(int),
        }
    ).reset_index(drop=True)
