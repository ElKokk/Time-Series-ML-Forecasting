"""End-to-end regression test for the forecasting pipeline.

Runs train_and_forecast against the bundled synthetic data with a fixed,
small Optuna budget and asserts the resulting metrics match a canonical
reference run. Because the Optuna TPE sampler is seeded and XGBoost is
deterministic on CPU with random_state=42, the numbers below are
reproducible across machines.

This is the safety net. It exercises:

  - The hyperparameter search (n_trials=2)
  - Feature engineering on the real schema
  - The recursive history-rebuild loop (forecast_weeks=2)
  - The metric reporting that the dashboard ultimately consumes

Any future refactor that silently changes the modeling math will fail
this test even when the more focused unit tests still pass.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from forecast import train_and_forecast


SAMPLE_DATA_CSV = (
    Path(__file__).resolve().parent.parent
    / "sample_data"
    / "combined_for_model2.csv"
)

# Captured from a clean reference run with the seeded sampler.
# Update these values *only* if a refactor is intentionally changing the
# modeling math, and document the reason in the commit message.
CANONICAL_MAE = 2758.72
CANONICAL_RMSE = 3548.54
CANONICAL_MAPE = 3.74
CANONICAL_R2 = 0.9105


def test_end_to_end_regression_matches_canonical_metrics(tmp_path):
    df = pd.read_csv(SAMPLE_DATA_CSV, index_col=0)

    results = train_and_forecast(
        df=df,
        response_variable="Dry Actuals",
        n_trials=2,
        output_dir=str(tmp_path),
        n_bootstraps=500,
        ci_level=95,
        forecast_weeks=2,
        device="cpu",
    )

    metrics = results["model"]
    assert metrics["mae"] == pytest.approx(CANONICAL_MAE, abs=0.01)
    assert metrics["rmse"] == pytest.approx(CANONICAL_RMSE, abs=0.01)
    assert metrics["mape"] == pytest.approx(CANONICAL_MAPE, abs=0.01)
    assert metrics["r2"] == pytest.approx(CANONICAL_R2, abs=0.001)

    # Two weeks of six business days each = 12 prediction rows.
    predictions_df = results["predictions"]
    assert len(predictions_df) == 12
    assert {"Actual", "Model_Prediction", "Existing_Forecast"} <= set(
        predictions_df.columns
    )
