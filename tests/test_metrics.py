"""Numerical correctness of the regression metrics in forecast.compute_metrics.

The metrics returned by ``compute_metrics`` flow into every printed result,
the bootstrap confidence intervals, the dashboard's headline table and the
final return value of ``train_and_forecast``. If the math here is wrong then
every other test that asserts on metric values is wrong by the same amount,
so this test exists to pin it down against hand-computed values that don't
go through scikit-learn.
"""

from __future__ import annotations

import numpy as np
import pytest

from forecast import compute_metrics


def test_compute_metrics_against_hand_calculated_values():
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([90.0, 210.0, 280.0, 420.0])

    mae, rmse, mape, r2 = compute_metrics(y_true, y_pred)

    # MAE = mean(|y_true - y_pred|)
    #     = mean(|100-90|, |200-210|, |300-280|, |400-420|)
    #     = mean(10, 10, 20, 20) = 15
    assert mae == pytest.approx(15.0)

    # RMSE = sqrt(mean((y_true - y_pred)^2))
    #      = sqrt(mean(100, 100, 400, 400))
    #      = sqrt(250)
    assert rmse == pytest.approx(np.sqrt(250.0))

    # MAPE = mean(|(y_true - y_pred) / y_true|) * 100
    #      = mean(0.10, 0.05, 0.0666..., 0.05) * 100
    #      = 0.066666... * 100
    expected_mape = (0.10 + 0.05 + (20.0 / 300.0) + 0.05) / 4 * 100
    assert mape == pytest.approx(expected_mape)

    # R^2 = 1 - SS_res / SS_tot
    # SS_res = 100 + 100 + 400 + 400 = 1000
    # mean(y_true) = 250 ; SS_tot = 150^2 + 50^2 + 50^2 + 150^2 = 50_000
    expected_r2 = 1.0 - 1000.0 / 50_000.0
    assert r2 == pytest.approx(expected_r2)


def test_compute_metrics_handles_perfect_predictions():
    """A perfect predictor returns MAE=0, RMSE=0, MAPE=0, R2=1."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mae, rmse, mape, r2 = compute_metrics(y_true, y_true)

    assert mae == 0.0
    assert rmse == 0.0
    assert mape == 0.0
    assert r2 == pytest.approx(1.0)
