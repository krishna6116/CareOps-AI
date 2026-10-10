# CareOps AI — Hospital Operations Forecasting and Resource Planning

A decision-support portfolio project. It forecasts hospital admissions, occupied
beds, ICU admissions, length of stay and appointment no-shows, then uses Google
OR-Tools to plan beds, staff shifts and appointments against those forecasts.
A FastAPI service and a React dashboard present the results.

> **Synthetic data only.** Every patient, bed, shift and appointment is
> generated. Nothing here has been validated on a real hospital, and the
> outputs are not clinical or staffing decisions. See
> [Limitations](#limitations-and-synthetic-data).

## The problem

Bed managers and staffing coordinators plan days to weeks ahead with uncertain
demand. Too little capacity means boarding and cancelled procedures; too much
wastes staff hours. This project asks two questions:

1. **How well can we forecast demand 30 days ahead**, and does a machine-learning
   model actually beat simple rules of thumb?
2. **Given those forecasts, what bed, staff and appointment plan satisfies the
   operational constraints**, and what extra capacity would a feasible plan need?

## Architecture

```mermaid
flowchart LR
  G[Synthetic data generator<br/>data/synthetic] --> V{Source integrity checks}
  V --> A[Admissions forecast<br/>XGBoost vs naive baselines]
  V --> L[Length-of-stay model]
  V --> O[Occupancy forecast<br/>Ridge vs naive baselines]
  V --> I[ICU admissions forecast]
  V --> N[No-show model]
  A & L & O & I --> B[Bed availability projection<br/>whole beds]
  B --> R[OR-Tools CP-SAT<br/>beds, staff, appointments]
  N --> R
  R --> S[Reliability scorecard<br/>+ run manifest]
  S --> M[docs/MODEL_METRICS.md]
  S --> API[FastAPI] --> UI[React dashboard]
  S -.optional.-> PG[(PostgreSQL)]
  V -.--exploratory.-> E[Cleaning, EDA,<br/>feature tables]
```

Each forecast is published only after a 30-day rolling-origin backtest against
naive baselines. If the learned model does not win, the baseline forecast is
published.

## Results

All numbers come from `run_portfolio.py --as-of-date 2026-10-05` and are
regenerated into [docs/MODEL_METRICS.md](docs/MODEL_METRICS.md) on every run.

### Forecast accuracy (model results)

30-day rolling-origin backtest: 12 origins, 30 days each, inside a held-out final
20% of history. Learned models forecast recursively, as they do in production.
MASE below 1 means better than the same-weekday naive forecast.

| Target | Published model | MAE | Best naive baseline | Learned model | Verdict |
|---|---|---|---|---|---|
| Hospital admissions / day (~11.5 avg) | 28-day moving average | 2.94 | 28-day moving average 2.94 | XGBoost 3.20 | **XGBoost does not beat the baseline** |
| Occupied beds (of 230) | Ridge, recursive | 9.17 | 28-day moving average 12.0 | Ridge 9.17 (MASE 0.91) | Beats every baseline |

Interval coverage on the later backtest origins (target 80% / 95%):
admissions 67–79% / 88–90%; occupied beds 74–86% / 88–98%. Intervals are
somewhat too narrow for the first week of the horizon.

One-step chronological holdouts (not yet 30-day backtested): ICU admissions MAE
0.73/day vs 1.01 for the 7-day baseline; length of stay MAE 2.59 days; no-show
Brier 0.113 vs 0.128 for the prevalence baseline, ROC-AUC 0.74.

**Why admissions don't beat the baseline:** in this synthetic data, daily
admissions behave like Poisson noise around a stable level (variance ≈ mean).
There is little structure beyond the average for any model to learn.
Occupancy is different: it is a stock that changes slowly, so recent history
carries real information.

### Planning results (simulation, not model accuracy)

The optimizer finds a plan with zero bed, ICU and emergency overflow and no
unfilled shifts. It needs **scenario-only** extra capacity to do so: temporary
surge beds and a synthetic flex-staff pool, both listed in
`optimization/resource_planning/outputs/capacity_expansion_plan.csv`. These
show the plan is feasible under the model's assumptions. They are not
measured improvements, and this project makes no claim about real-world
savings or outcomes.

## How to run

Requirements: Python 3.12, and Node 20+ for the dashboard.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

# Full pipeline: data → forecasts → optimization → scorecard → docs/MODEL_METRICS.md
.\.venv\Scripts\python.exe run_portfolio.py --as-of-date 2026-10-05

# Also rebuild the cleaned data, EDA reports and exploratory feature tables
.\.venv\Scripts\python.exe run_portfolio.py --as-of-date 2026-10-05 --exploratory

# Tests
.\.venv\Scripts\python.exe -m pytest -q
```

Generators and models use fixed seeds. The same date, code and dependency
versions reproduce the same data and forecasts. `requirements-lock.txt` records
the exact package versions behind the committed outputs. A run exits nonzero and lists
its blockers in `portfolio_run_manifest.json` if any reliability section is
below 4/5 or a capacity gate fails. The default `--as-of-date` is today; the
committed outputs are for 2026-10-05.

### API and dashboard

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
# API docs: http://localhost:8000/docs

Set-Location frontend
npm ci
$env:VITE_API_URL = "http://localhost:8000"
npm run dev   # http://localhost:5173
```

The API serves the saved pipeline outputs. SQLite is used by default for alerts
and simulation records.

### Docker

```powershell
Copy-Item .env.example .env   # then set a unique POSTGRES_PASSWORD
docker compose up --build
```

### PostgreSQL handoff (optional)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-postgresql.txt
$env:CAREOPS_DATABASE_URL = "postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE"
.\.venv\Scripts\python.exe run_portfolio.py --as-of-date 2026-10-05 --postgres
Remove-Item Env:\CAREOPS_DATABASE_URL
```

This loads source data, forecasts and optimization outputs into a `careops`
schema in one transaction, tagged by `run_id`.

## Forecasting rules

- **Time-based splits only.** No shuffling. Lags and rolling features use data
  up to the previous day.
- **Naive baselines everywhere:** last value, same weekday last week, 7- and
  28-day moving averages. Metrics: MAE, RMSE, sMAPE, MASE and bias.
- **Whole numbers at the output layer.** Admissions, beds and ICU admissions are
  published as non-negative integers, and occupied beds are capped at bed
  capacity (`ml/count_outputs.py`, covered by unit tests). Department
  admissions are reconciled so they add up to the hospital total. Evaluation
  files keep raw model output for scoring.
- **Prediction intervals** are split-conformal, use the backtest errors, and are
  reported with their measured coverage.

## Project layout

```text
run_portfolio.py              One-command pipeline (+ --exploratory, --postgres)
reliability_scorecard.py      Evidence-based section scores for each run
data/
  synthetic/                  Generated source data (generate_data.py) + resource_support/
  data_cleaning.py            → data/processed/          (exploratory)
  eda_complete.py             → data/eda/                 (exploratory)
  feature_engineering.py      → data/feature_engineering/ (exploratory, not committed)
ml/
  count_outputs.py            Integer / non-negative / capacity rules
  forecast_evaluation.py      Baselines, rolling-origin backtest, intervals
  model_metrics_report.py     Writes docs/MODEL_METRICS.md
  admission_forecasting.py    + admission_forecasting/ outputs and charts
  occupancy_forecasting.py    + occupancy_forecasting/ outputs
  icu_demand_forecasting/, los_prediction/, no_show_prediction/
optimization/
  resource_optimization.py    OR-Tools planner (beds, staff, appointments, emergency)
  resource_validation.py      Input audit and holdout scoring
  bed_allocation/             Standalone bed-allocation study
  resource_planning/outputs/  Planner outputs, scorecard, run manifest
backend/app/                  FastAPI service
frontend/                     React + TypeScript dashboard
tests/                        Pipeline, evaluation and output-rule tests
docs/                         Model metrics, architecture, API, deployment notes
```

The forecasting models build their own leakage-safe features from
`data/synthetic`. The exploratory feature tables are for analysis and are not
model inputs.

## Limitations and synthetic data

- **Synthetic data.** The generator encodes its own assumptions about arrival
  rates, seasonality, length of stay and no-show behaviour. Models that do well
  here are learning the generator, not a hospital. Real data would bring
  missing values, coding changes, reporting delays and shocks that are absent here.
- **Simulation warm-up.** The simulated hospital starts empty on 2024-01-01;
  occupancy ramps up over the first weeks. Those days are included in training
  history.
- **Retrospective evaluation only.** Backtests use 12 origins on one simulated
  history. That is not a prospective trial, and interval coverage estimates
  from 6 evaluation origins are noisy.
- **Admissions are near noise.** The published admissions forecast is a 28-day
  moving average because nothing tested beat it.
- **ICU, LOS and no-show** models have one-step holdout checks only. The ICU
  model forecasts admissions, not ICU census.
- **Planning feasibility depends on scenario capacity.** Zero overflow is
  achieved with synthetic surge beds and flex staff. Costs are illustrative
  assumptions, not quotes.
- **Whole-number rounding.** Published bed-projection columns are rounded one
  by one, so `current + admissions − discharges` may differ from the projected
  beds by a unit.
- **Not for clinical use.** The no-show score must not be used to deny or
  deprioritise care. All outputs are intended for review by qualified staff.

More detail: [model metrics](docs/MODEL_METRICS.md),
[resource optimization](docs/RESOURCE_OPTIMIZATION.md),
[architecture](docs/architecture.md), [API](docs/api.md),
[deployment](docs/deployment.md), [AI assistant](docs/ai_assistant.md).
