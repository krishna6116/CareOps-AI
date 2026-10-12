# API documentation

The FastAPI service exposes the synthetic source data and the saved model and
optimization outputs. Missing files return 404 or 503; they are never
replaced with made-up values. Interactive OpenAPI documentation is at `/docs`
and `/redoc`. CORS allows only the origins in `CAREOPS_CORS_ORIGINS`.

## Service

- `GET /health`: liveness and database status (`engine`: `sqlite` or `postgresql`). Never returns connection strings or driver errors.
- `GET /ready`: 200 when the database is reachable and a completed pipeline run exists, otherwise 503 with the failing check.
- `GET /meta`: app version, environment, data scope, scenario date, horizon, pipeline status and assistant provider.

## Source data and datasets

- `GET /patients`, `GET /patients/{patient_id}`, `GET /beds`, `GET /beds/availability`, `GET /admissions`
- `GET /datasets?search=&area=data|ml|optimization`: catalog of generated CSVs with schema, row counts and date ranges.
- `GET /datasets/{dataset_id}?offset=&limit=`: paginated rows (limit ≤ 500). IDs are encoded paths confined to `data/`, `ml/` and `optimization/`.
- `GET /validation/datasets`: quality checks per dataset.
- `GET /departments`, `GET /operations?start_date=&end_date=&department=`

## Forecasts and models

- `GET /forecast/history?days=90`: historical daily operations, all departments.
- `GET|POST /forecast/admissions`, `GET|POST /forecast/occupancy`: saved forecasts; `POST` filters from `as_of_date`.
- `GET /forecast/icu`: ICU admissions per day, not ICU census.
- `GET /los/predictions?limit=500`, `POST /los/predict`: saved LOS predictions filtered by `patient_id` and/or `department`.
- `GET /no-show/predictions?limit=500`: highest-risk predictions first (limit ≤ 5000).
- `GET /no-show/summary`: risk-band counts over all predictions, observed no-show rate and window.
- `GET /models/metrics`: raw model metadata files.
- `GET /models/evaluation`: published model vs best baseline and best learned model, interval coverage and occupancy alert thresholds. This is the same summary as `docs/MODEL_METRICS.md`.

## Dashboard summaries

`GET /dashboard/overview`, `/dashboard/beds`, `/dashboard/staff`,
`/dashboard/appointments` (top 500 no-show rows) and `/dashboard/optimization`.

## Optimization

- `POST /optimization/run`: runs the whole pipeline. Long-running; intended for local use.
- `GET /optimization/runs`, `GET /optimization/runs/{run_id}`, `GET /optimization/runs/{run_id}/allocations`
- `GET /optimization/bed-assignments`: request-level replay summary covering window, policies, objective weights, constraints, solver statuses, KPIs, reason definitions, the daily solver log and the sensitivity table.
- `GET /optimization/bed-assignments/requests?policy=baseline|optimized&status=ASSIGNED|UNASSIGNED&planning_date=&department=&offset=&limit=`: one row per request, with both policies' beds and reason codes (limit ≤ 500).

## Recommendations, alerts, simulation, assistant, audit

- `GET /recommendations`, `POST /recommendations/generate`
- `GET /alerts`, `POST /alerts` (validated; persisted)
- `POST /simulation/run`: admissions multiplier 0–3, ICU bed change −20 to 20, staff availability change −1 to 1. Reruns the CP-SAT department allocator and emergency model on adjusted copies of the saved inputs and persists the run. Count metrics are integers.
- `GET /simulation/runs`, `GET /simulation/{simulation_id}`
- `POST /assistant/ask`: grounded answer with evidence; see [ai_assistant.md](./ai_assistant.md).
- `GET /audit/runs` (data-quality audit), `GET /audit/assistant?limit=100`

Request schemas reject unknown fields where a silently ignored filter would
mislead. Validation errors return 422. The main routes declare Pydantic
response models, so `/docs` shows their exact shape.

## Access control

- **Admin endpoints** (`POST /optimization/run`, `POST /alerts`,
  `POST /recommendations/generate`) require `X-API-Key` to equal
  `CAREOPS_API_KEY`. A wrong key returns 401. If no key is configured, these
  endpoints work only when `CAREOPS_ENVIRONMENT=development` and otherwise
  return 503, so a misconfigured deployment fails closed.
- **Interactive solver endpoints** (`POST /simulation/run`,
  `POST /assistant/ask`) stay open for the dashboard. They are limited per
  client address to `CAREOPS_INTERACTIVE_RATE_LIMIT_PER_MINUTE` (default 20)
  and return 429 with `Retry-After`. The limiter is per process.
- Read endpoints are public and serve synthetic data only.
