"""Generate a small synthetic dataset for the forecasting pipeline.

The real Delhaize inbound dataset cannot be shared, so this script produces a
schema-compatible substitute that lets reviewers run ``src/forecast.py`` and
``src/dashboard.py`` end to end. The signal is intentionally simple: a slow
upward trend, a weekly seasonal pattern, and gaussian noise on top.

Outputs (written to ``sample_data/`` by default):
    combined_for_model2.csv      input for forecast.py and dashboard.py
    Merged_Predictions_Data.csv  pre-baked predictions for the dashboard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Actuals -> existing-forecast column name, matching forecast.py.
CATEGORY_TO_FC = {
    "Dry Actuals": "Dry Fc",
    "Fresh": "Fresh Fc",
    "Ultrafresh": "Ultrafresh Fc",
    "Frozen": "Frozen Fc",
}
CATEGORIES = list(CATEGORY_TO_FC.keys())
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def _business_days(start: str, n_days: int) -> pd.DatetimeIndex:
    """Return ``n_days`` consecutive Mon-Sat dates starting from ``start``."""
    dates = pd.date_range(start=start, periods=n_days * 2, freq="D")
    return dates[dates.dayofweek < 6][:n_days]


def _seasonal_series(n: int, base: float, amplitude: float, noise: float,
                     rng: np.random.Generator) -> np.ndarray:
    """Trend + weekly seasonality + noise."""
    t = np.arange(n)
    trend = base + 0.05 * t
    seasonal = amplitude * np.sin(2 * np.pi * t / 6)
    noise_term = rng.normal(0, noise, size=n)
    return np.maximum(trend + seasonal + noise_term, 0)


def build_actuals(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = _business_days("2023-01-02", n_days)

    df = pd.DataFrame({"Date": dates})
    df["Day of Week"] = df["Date"].dt.day_name()
    df["Category"] = "All"

    # Actuals + a slightly noisier "existing forecast" baseline per category.
    for cat, base, amp, noise in [
        ("Dry Actuals", 80_000, 15_000, 4_000),
        ("Fresh",       60_000, 12_000, 3_000),
        ("Ultrafresh",   8_000,  2_000,   600),
        ("Frozen",       7_000,  1_500,   500),
    ]:
        df[cat] = _seasonal_series(n_days, base, amp, noise, rng).round()
        df[CATEGORY_TO_FC[cat]] = (
            df[cat] + rng.normal(0, noise * 1.5, n_days)
        ).round()

    df["Total Inbound"] = df[CATEGORIES].sum(axis=1)
    df["Total Inbound Fc"] = (
        df["Total Inbound"] + rng.normal(0, 6_000, n_days)
    ).round()
    df["Pre-orders Fc"] = (df["Total Inbound"] * 0.15).round()

    # A few promo and product side-features so product_cols is non-empty.
    df["Promo Dry"] = rng.integers(0, 5, n_days)
    df["Promo Fresh"] = rng.integers(0, 5, n_days)
    df["Promo Frozen"] = rng.integers(0, 3, n_days)
    df["HD Total"] = rng.integers(800, 1_400, n_days)
    df["Pick Capacity"] = rng.integers(10_000, 14_000, n_days)
    df["Collect Total"] = rng.integers(500, 1_200, n_days)
    df["Pre Orders"] = (df["Total Inbound"] * 0.12).round()

    return df


def build_predictions(actuals: pd.DataFrame, weeks: int, seed: int) -> pd.DataFrame:
    """Synthesize the merged-predictions file the dashboard expects."""
    rng = np.random.default_rng(seed + 1)
    tail = actuals.tail(weeks * 6).copy().reset_index(drop=True)
    out = pd.DataFrame({"Date": tail["Date"]})
    fc_lookup = {**CATEGORY_TO_FC, "Total Inbound": "Total Inbound Fc"}
    for cat in CATEGORIES + ["Total Inbound"]:
        out[f"Model_Prediction_{cat}"] = (
            tail[cat] + rng.normal(0, tail[cat].std() * 0.05, len(tail))
        ).round()
        out[f"Existing_Forecast_{cat}"] = tail[fc_lookup[cat]]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=600,
                        help="Number of business days to generate (default 600).")
    parser.add_argument("--forecast-weeks", type=int, default=3,
                        help="Number of weeks the prediction file should cover.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("sample_data"),
                        help="Output directory (default ./sample_data).")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    actuals = build_actuals(args.days, args.seed)
    predictions = build_predictions(actuals, args.forecast_weeks, args.seed)

    actuals_path = args.out / "combined_for_model2.csv"
    predictions_path = args.out / "Merged_Predictions_Data.csv"

    # forecast.py reads with index_col=0, so we write a row index.
    actuals.to_csv(actuals_path, index=True)
    predictions.to_csv(predictions_path, index=False)

    print(f"Wrote {len(actuals)} rows to {actuals_path}")
    print(f"Wrote {len(predictions)} rows to {predictions_path}")


if __name__ == "__main__":
    main()
