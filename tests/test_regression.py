"""End-to-end regression test for the forecasting pipeline.

Runs train_and_forecast against the bundled synthetic data with a fixed,
small Optuna budget and asserts the resulting metrics fall inside a sensible
band. The bounds are wide enough to survive XGBoost's small floating-point
drift between Python versions and BLAS backends, and tight enough that any
real refactoring bug (a broken fold-back, a wrong feature column, target
leakage) would push the metrics well outside them.

For reference, here are the values produced by two known environments:

    Python 3.13 + xgboost 3.2.0 (local):
        MAE 2758.72   RMSE 3548.54   MAPE 3.74%   R2 0.9105
    Python 3.12 + xgboost (CI runner):
        MAE 2592.74   ...

Both sit comfortably inside the bounds below. If a future refactor pushes
the metrics out of these ranges, that is exactly the signal this test exists
to surface — investigate before widening the bounds.

This test exercises the hyperparameter search, feature engineering on the
real schema, the recursive history-rebuild loop (forecast_weeks=2) and the
metric reporting the dashboard ultimately consumes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from forecast import train_and_forecast


SAMPLE_DATA_CSV = (
    Path(__file__).resolve().parent.parent
    / "sample_data"
    / "combined_for_model2.csv"
)


def test_end_to_end_regression_produces_reasonable_metrics(tmp_path):
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
    assert 1500 < metrics["mae"] < 4000, f"MAE out of band: {metrics['mae']}"
    assert 2000 < metrics["rmse"] < 5000, f"RMSE out of band: {metrics['rmse']}"
    assert 2.0 < metrics["mape"] < 6.0, f"MAPE out of band: {metrics['mape']}"
    assert 0.85 < metrics["r2"] < 0.99, f"R2 out of band: {metrics['r2']}"

    # Two weeks of six business days each = 12 prediction rows.
    predictions_df = results["predictions"]
    assert len(predictions_df) == 12
    assert {"Actual", "Model_Prediction", "Existing_Forecast"} <= set(
        predictions_df.columns
    )
