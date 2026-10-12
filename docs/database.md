# Database design and PostgreSQL

## Two kinds of storage

1. **Application tables.** These are defined in `backend/app/models/careops.py`
   and created by Alembic (`backend/alembic/versions`). There are 20 tables:
   - Entities: patients, beds, staff, admissions, discharges, appointments, shifts.
   - Forecasts: admission_forecasts, occupancy_forecasts, los_predictions,
     bed_availability.
   - Planning: bed_allocations, optimization_runs, alerts, recommendations.
   - Simulation: simulations and simulation_results. A simulation row stores
     its assumptions, baseline metrics and scenario metrics.
   - History: model_runs, audit_logs, assistant_audit_logs.

   Primary keys, foreign keys (for example admissions → patients and beds,
   simulation_results → simulations) and timestamps are declared on the
   models. CI applies the migrations to an empty database and runs
   `alembic check`, so the models and migrations cannot drift apart.
2. **Pipeline handoff.** `run_portfolio.py --postgres` calls
   `optimization/postgresql_integration.py`. It loads every source dataset,
   forecast, optimization output (including the request-level bed
   assignment) and quality report into a `careops` schema in one transaction.
   Every row is tagged with `run_id`, and a `pipeline_runs` table records each
   run. Reloading a run deletes and replaces its rows, so reruns do not
   duplicate data. Any error rolls back the whole load.

The API's read endpoints serve the canonical CSV/JSON outputs. They do not
claim to query PostgreSQL. What the API writes goes to the configured
database: alerts, simulations and the assistant audit.

## Configuration

- `CAREOPS_DATABASE_URL`: SQLite by default (`sqlite:///./careops.db`);
  PostgreSQL uses `postgresql+psycopg://USER:PASSWORD@HOST:5432/DB`. Never
  commit credentials; `.env` is git-ignored.
- PostgreSQL connections use a pool (`CAREOPS_DATABASE_POOL_SIZE`, default 5;
  `CAREOPS_DATABASE_MAX_OVERFLOW`, default 5) with pre-ping and 30-minute
  recycling, so a database restart does not fail the next request.
- Sessions are opened per operation and committed explicitly. A failed commit
  is rolled back.

## Setup

```powershell
$env:CAREOPS_DATABASE_URL = "postgresql+psycopg://USER:PASSWORD@localhost:5432/careops"
.\.venv\Scripts\python.exe -m alembic -c backend/alembic.ini upgrade head
```

Docker Compose runs the same migration before starting the API.

## Verification

`backend/app/tests/test_postgres_integration.py` runs when
`CAREOPS_TEST_DATABASE_URL` is set. It checks that:

- Alembic creates every model table on PostgreSQL.
- Synthetic records can be inserted and queried, and a foreign-key violation
  rolls back.
- A 30-day forecast is saved and read back unchanged.
- A simulation and an assistant audit record can be read through a new
  engine, as a restarted application would.
- The pipeline handoff loads every output with matching row counts, and
  reloading is idempotent.
- `/health` reports `postgresql`.

On 2026-10-12 all 6 tests passed locally against PostgreSQL 17.6. Separately,
an API process was stopped and a new one started against the same database,
and it still returned the simulation and assistant history written before the
restart. CI runs these tests against a PostgreSQL 16 service and fails if they
are skipped.
