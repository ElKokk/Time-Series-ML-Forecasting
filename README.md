# Delhaize Inbound Forecasting

A short-horizon forecasting pipeline for daily inbound volumes at a Delhaize
distribution centre. The model predicts one week of inbound with a **two-week
lead time** — i.e. on day *T* it forecasts days *T+14* through *T+19* — so
operations have enough notice to plan staffing, dock slots and capacity.

The pipeline is built around a recursive XGBoost regressor, hyperparameter
search with Optuna on time-series cross-validation folds, bootstrap confidence
intervals on every metric, and SHAP explanations exported per day. Performance
is benchmarked against the existing in-house forecast that the operations team
relies on today.

## Headline results

Mean Absolute Percentage Error on the held-out forecast horizon, model vs. the
existing forecast in production:

| Category   | Model MAPE | Existing MAPE | Δ accuracy |
|------------|-----------:|--------------:|-----------:|
| Dry        |     9.83 % |       12.38 % |   **+2.55 %** |
| Frozen     |    17.17 % |       21.12 % |   **+3.95 %** |
| Ultrafresh |    23.38 % |       26.08 % |   **+2.70 %** |
| Fresh      |     6.42 % |        5.06 % |       −1.36 % |

The model wins on three of the four product groups. Fresh is the one category
where the existing baseline is still slightly better — see the *Limitations*
section below for the reasons.

## Repository layout

```
.
├── src/
│   ├── forecast.py             # training + recursive forecasting CLI
│   └── dashboard.py            # Streamlit dashboard
├── tests/                      # pytest suite, see "Tests" below
│   ├── conftest.py
│   ├── test_metrics.py
│   ├── test_features.py
│   ├── test_predict_week.py
│   └── test_regression.py
├── scripts/
│   └── generate_synthetic_data.py
├── sample_data/                # synthetic CSVs so the project is runnable
│   ├── combined_for_model2.csv
│   ├── Merged_Predictions_Data.csv
│   └── headline_metrics.csv
├── docs/
│   └── final_presentation.pptx
├── pyproject.toml              # pytest config
├── requirements.txt
├── LICENSE
└── README.md
```

The real Delhaize dataset is confidential and is **not** included. The CSVs
under `sample_data/` are produced by `scripts/generate_synthetic_data.py` and
follow the exact same schema, so every command below works out of the box.

## Quick start

```bash
# 1. Install dependencies (Python 3.9+ recommended)
pip install -r requirements.txt

# 2. (Optional) regenerate the synthetic dataset
python scripts/generate_synthetic_data.py

# 3. Train and forecast — small run for a smoke test
python src/forecast.py \
    --dataset sample_data/combined_for_model2.csv \
    --response "Dry Actuals" \
    --epochs 1 \
    --forecast_weeks 1 \
    --output_dir results/

# 4. Launch the dashboard
streamlit run src/dashboard.py
```

A realistic run uses `--epochs 50 --forecast_weeks 3`. The script writes the
predictions, the actual-vs-predicted plot, the SHAP beeswarm, per-day SHAP
force plots, feature importances and the bootstrap confidence intervals into
`--output_dir`.

### Forecast CLI arguments

| Flag               | Default | Description                                          |
|--------------------|--------:|------------------------------------------------------|
| `--dataset`        |       — | Path to the input CSV.                               |
| `--response`       |       — | Target column (`Dry Actuals`, `Fresh`, `Ultrafresh`, `Frozen`, `Total Inbound`). |
| `--epochs`         |      50 | Number of Optuna trials.                             |
| `--forecast_weeks` |       3 | Number of weeks to roll forward through.             |
| `--output_dir`     |     `.` | Where artefacts (CSVs, plots, SHAP) are written.     |
| `--device`         |   `cpu` | XGBoost device. Pass `cuda` to use a GPU build.      |

## How the model works

A few things worth highlighting if you read `src/forecast.py`:

- **Recursive multi-week forecasting.** Each week is trained on everything
  available up to its start, predicted, and then folded back into the
  history before training the next week. The lag features `[6, 7, 8, 14,
  16, 21]` are all at least one work-week long, so no row uses a feature
  derived from another row in the same forecast week. Smaller lags reach
  into either real history (for week 1) or the model's own week-1 / week-2
  predictions (for weeks 2 and 3) — never a real future value.
- **Feature engineering.** Polynomial time index, Fourier seasonality
  (period 6, order 3), per-weekday rolling and EMA means, days-since-last-peak,
  a two-peak weekly pattern, and interactions between lagged response values
  and `Day_Monday`, ISO week, and `days_since_last_peak`.
- **Optuna over time-series CV.** Hyperparameters are searched with
  `optuna.create_study(direction="minimize")` evaluated through a 5-split
  `TimeSeriesSplit` so the search never leaks future information.
- **Bootstrap confidence intervals.** MAE, RMSE and MAPE each get a 95% CI
  from 5 000 bootstrap resamples, which makes the comparison against the
  existing forecast meaningful instead of a single-number coin toss.
- **SHAP explanations per forecast day.** A summary beeswarm plus a force
  plot for each predicted day, plus the raw SHAP values exported as CSV so
  the operations team can audit individual predictions.

## Dashboard

`src/dashboard.py` is a small Streamlit app that loads the inbound CSV and
the merged predictions CSV and exposes:

- An overview of the actuals and the forecast horizon.
- Trend analysis with toggleable categories.
- Promo activity overlays.
- Summary statistics, histograms and box plots per category.
- Time-series decomposition (additive, period 7) for any selected metric.
- A model-vs-existing performance table.

From a clean checkout it just works against the bundled sample data:

```bash
streamlit run src/dashboard.py
```

The dashboard looks for `combined_for_model2.csv`, `Merged_Predictions_Data.csv`
and `headline_metrics.csv` in the current working directory first, and falls
back to `sample_data/` if they are not there. To run it against real data,
launch streamlit from a directory that contains your own copies of those
files.

## Tests

The project ships with a small pytest suite that pins the contracts the
forecasting pipeline depends on. Run it from the repo root:

```bash
pytest
```

Seven tests across four files, ~90 seconds end to end (most of which is the
seeded regression test running an actual Optuna search):

| File | What it pins |
|---|---|
| `tests/test_metrics.py` | MAE, RMSE, MAPE and R² math against hand-computed values, plus a perfect-predictor sanity check. |
| `tests/test_features.py` | Lag-column offsets are exact (lag_k at row r equals source[r − k]) and the smallest lag offset is at least 6, so no row reaches into its own forecast week. |
| `tests/test_predict_week.py` | The `_predict_week` helper zero-fills missing columns and reorders to the booster's column order, and handles single-row inputs. |
| `tests/test_regression.py` | End-to-end seeded run of `train_and_forecast` against `sample_data/` produces canonical MAE/RMSE/MAPE/R² values. Exercises the recursive history-rebuild loop. |

The regression test exists specifically as a safety net for refactoring the
recursive forecast loop and the `create_features` function — both have
intricate state coupling that the unit tests cannot catch on their own. If a
future change is intentionally moving the modeling math, update the
`CANONICAL_*` values in `tests/test_regression.py` and document the reason in
the commit message.

## Limitations and next steps

A few things I would change or extend if this were continued:

- **Fresh is still hard.** The existing baseline beats the model by 1.36 pp on
  Fresh, and the wide MAPE confidence interval on Ultrafresh
  (`[12.03 %, 40.30 %]`) shows how thin the signal gets on the more volatile
  categories. Both would benefit from more granular promo and weather
  features.
- **The recursive loop rebuilds features in-place.** It works, but it would be
  cleaner to factor the feature pipeline into a `sklearn` transformer and run
  it inside a `Pipeline` so that train/test contamination is impossible by
  construction.
- **No persisted model.** Each run retrains from scratch. For a real
  deployment the best Optuna trial would be serialized and reloaded.

## Author

Eleftherios Kokkinis

## License

MIT — see [LICENSE](LICENSE).
