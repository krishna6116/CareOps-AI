# CareOps AI — Hospital Resource Planning

An operations-research portfolio project for forecasting bed demand and planning
beds, clinical staffing, appointments, and emergency capacity with Python,
pandas, and Google OR-Tools CP-SAT.

> **Scope:** This repository contains a synthetic-data planning simulation. It
> is not a clinical decision system, and its outputs are not deployment-ready
> staffing orders or patient-placement recommendations.

## What it does

- Combines admission, length-of-stay, current-occupancy, ICU, and hospital
  occupancy forecasts into a 30-day department bed-demand forecast.
- Selects admission model complexity per department using a chronological
  validation slice, then reports untouched holdout performance against a
  rolling one-week seasonal baseline.
- Compares a department-first baseline with CP-SAT bed allocation under
  department/type compatibility, staffing caps, protected reserves, transfers,
  and emergency-flexible beds.
- Schedules available staff by role and department while respecting shift
  availability, weekly hours, at-most-one-shift-per-day, and minimum rest;
  provider shifts prioritize department-level appointment coverage.
- Reassigns appointments on the latest appointment day to available
  specialty-matched providers, avoiding doctor-time overlap and considering
  urgency, duration, and shift availability. No-show scores are retained for
  review and are not currently used to move appointments.
- Compares ICU and emergency demand with shared bed allocations and staffing.
- Writes a daily resource-gap analysis that distinguishes staffing limits from
  dedicated bed/compatibility limits.
- Audits input freshness, identifier/grain uniqueness, date consistency, and
  invalid forecast values before optimization.
- Scores saved chronological holdout predictions against simple baselines and
  reports model performance by department where available.
- Fits validation-only probability calibration for no-show scores and writes an
  evidence-based reliability scorecard for every portfolio section.
- Generates a reproducible support bundle with dated synthetic bed/staff
  snapshots and explicitly labeled isolation-scenario labels/rates.
- Sizes scenario-only temporary beds and an agency-style flex roster against
  forecast peaks, then fails the run unless the optimized plan has zero unfilled
  shift positions, unassigned appointments, bed overflow, ICU overflow, and
  emergency overflow.

## Reproducible scenario

The generated KPI values are scenario outputs, not fixed claims. They are
written with each run to `optimization/resource_planning/outputs/` and
summarized in `resource_optimization_report.txt` and
`portfolio_run_manifest.json`. The original constrained baseline remains
visible; the optimized scenario may use explicit synthetic surge beds and
agency-style flex staffing. Their assumed quantities and illustrative costs
are recorded in `capacity_expansion_plan.csv` rather than treated as existing
hospital resources.

The support generator supplies synthetic isolation labels and department-rate
forecasts so the isolation constraints run end to end. These are scenario
assumptions, not clinical determinations or evidence of actual isolation need.
The ICU forecaster predicts daily ICU admissions, not occupied-bed census. The
resource planner uses its calibrated ICU occupancy projection for bed capacity
and reports ICU admissions as a separate flow signal. The model has no
patient-level queue data and does not report actual waiting time. Zero overflow
in the optimized scenario is conditional on modeled surge capacity being
available; it is not a claim about current or real-world hospital capacity.

The source generator calibrates daily admissions to the 230-bed inventory and
derives daily arrivals, discharges, ICU flows, occupied census, and the final
bed snapshot from the same encounter records. Historical bed assignments are
department/type compatible and non-overlapping. Admissions explicitly label
stays still open at the scenario cutoff; the LOS model trains only on completed
stays. Synthetic appointment outcomes are generated in appointment-time order,
with department-matched providers, and generated staff shifts respect rolling
weekly-hour limits and minimum rest.

## Run

From the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe run_portfolio.py
```

This rebuilds the base synthetic data in `data\synthetic\`, retrains each
forecast in dependency order, regenerates the resource support inputs, runs the
resource planner, and checks that every 30-day forecast covers the same dates.
The final handoff verifies the reliability scorecard, source-integrity checks,
capacity gates, and the leakage-safe No-show model metadata before marking the
manifest ready for a downstream process. A run with any section below target,
failed capacity gate, or No-show rating below 4.5/5 exits nonzero and records
the blockers in `portfolio_run_manifest.json`; it does not silently signal
readiness. Even a passing synthetic run is not approved for live clinical or
operational use. The existing `data\processed\` inputs are not overwritten.
The default scenario cutoff is today; pass an explicit date to reproduce a prior run:

```powershell
.\.venv\Scripts\python.exe run_portfolio.py --as-of-date 2026-09-29
```

The data/model generators use fixed seeds; for the same date, code, and
dependency versions the scenario is reproducible. Outputs are written beneath
`optimization/resource_planning/outputs/`.

### PostgreSQL handoff

Install the optional PostgreSQL dependencies and set the database URL in the
current PowerShell session; do not put credentials in project files:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-postgresql.txt
$env:CAREOPS_DATABASE_URL = "postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE"
.\.venv\Scripts\python.exe run_portfolio.py --postgres
Remove-Item Env:\CAREOPS_DATABASE_URL
```

The pipeline creates the dedicated `careops` schema, loads source/support data,
model forecasts and holdout predictions, and optimization/quality outputs in one
transaction. Each table is tagged with `run_id` (the scenario date); rerunning
that date replaces only its rows. `careops.pipeline_runs` records run status,
overall score, and per-table row counts. The pipeline fails the handoff and
exits nonzero if PostgreSQL is not configured or loading fails. This does not
change the synthetic-data/live-use limitations.

The run also scores saved holdout predictions; these are retrospective model
checks and are not proof of live or multi-week forecast accuracy. The source audit verifies cross-file identifiers, episode dates and LOS, bed
assignment overlap/type, provider matching, shift limits, daily event counts,
census, and snapshot reconciliation. The resource audit
fails on structural errors and forecast-horizon mismatches. It still marks
synthetic assumptions and the physical-vs-operational bed denominator for
review. The manifest records scenario-only surge capacity and illustrative cost
assumptions. To refuse synthetic or stale inputs as a production-readiness check, set
`$env:CAREOPS_REQUIRE_FRESH_DATA = "1"` before running. The bundled historical
synthetic scenario intentionally does not pass that gate.

Run focused tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_*.py -v
```

The end-to-end run also writes
`optimization/resource_planning/outputs/reliability_scorecard.{json,csv,txt}`.
It includes an overall portfolio score, per-section scores, individual scores
for generated datasets and forecast holdout groups, and a detailed
`reliability_details.csv`. Dataset scores measure structural integrity within
the synthetic scenario; forecast scores use retrospective chronological
holdouts. Neither means clinically validated or guarantees future accuracy.

See [the resource optimization notes](docs/RESOURCE_OPTIMIZATION.md) for
assumptions, limitations, input-audit checks, backtesting interpretation, and
output-file descriptions.

## Deployment-facing architecture

The project includes a deployment-facing layer that reuses the existing synthetic
pipeline rather than rewriting it. The backend is a FastAPI application under
`backend/app/`; the React + TypeScript dashboard reads generated artifacts via
that API.

```mermaid
flowchart LR
  A[Synthetic source data] --> B[Admission forecast]
  A --> C[LOS prediction]
  A --> D[Occupancy forecast]
  B --> E[Bed availability forecast]
  C --> E
  D --> E
  E --> F[OR-Tools allocation]
  F --> G[Recommendations and audit]
  B --> H[FastAPI]
  C --> H
  D --> H
  F --> H
  G --> H
  H --> I[React dashboard]
  H --> J[PostgreSQL handoff and simulation records]
```

Forecast and optimizer responses are backed by saved pipeline outputs. API reads
use generated CSV artifacts; PostgreSQL is validated separately for the
portfolio handoff and persisted alerts/simulation records. This project does
not claim live clinical performance.

### Run the API

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

The API docs are available at `http://localhost:8000/docs`; health is at
`http://localhost:8000/health`. SQLite is used by default. To run with PostgreSQL,
set `CAREOPS_DATABASE_URL` and `CAREOPS_CORS_ORIGINS` in the process environment.

### Run with Docker

```powershell
Copy-Item .env.example .env
# Edit .env and replace POSTGRES_PASSWORD with a unique, URL-safe local password.
docker compose up --build
```

Compose waits for PostgreSQL health, applies the Alembic schema migration, and
waits for backend health before serving the production React bundle from Nginx.
Docker must be installed and running for this option.

### Frontend dashboard

The multi-page React dashboard under `frontend/` covers overview, forecasting,
bed management, staff, appointments, optimization, alerts, simulation, the
AI assistant, audit/run history, and project limitations. OpenAI is the
primary cloud assistant provider; a rules-based demo mode is available without
a key. Server-side setup is documented in
`docs/ai_assistant.md`. Run the dashboard locally after
starting the backend:

Artifact tables include searchable, paginated views of all returned rows and
columns. The optimization page also exposes saved staffing and appointment
KPIs, capacity-expansion assumptions, resource gaps, forecast backtests, and
reliability scorecard evidence.

```powershell
Set-Location frontend
npm ci
$env:VITE_API_URL = "http://localhost:8000"
npm run dev
```

Open `http://localhost:5173`. Build and type-check with `npm run build`.

### Database and migration notes

The PostgreSQL layer uses SQLAlchemy models and Alembic configuration under
`backend/alembic/` and `backend/app/models/`. Set `CAREOPS_DATABASE_URL` to your
PostgreSQL instance before running in a production-like environment. The
portfolio handoff writes generated data to the `careops` schema; API reads are
from canonical generated CSV artifacts. SQLAlchemy persists alerts and
simulation records. Keep credentials outside source control.

### Validation and CI

Install project dependencies with `python -m pip install -r requirements-dev.txt`.
Run Python tests with `python -m unittest discover -s tests -p "test_*.py" -v`
and `python -m pytest backend/app/tests -q`. GitHub Actions runs those checks
plus the frontend's clean install and production build.
