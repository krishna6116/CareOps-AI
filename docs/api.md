# API documentation

The FastAPI service exposes saved synthetic source data and generated model and
optimization artifacts. Missing files are not replaced with generated demo
values. Interactive OpenAPI documentation is available at `/docs` and `/redoc`.

## Health and source data

- `GET /health`
- `GET /patients`, `GET /patients/{patient_id}`
- `GET /beds`, `GET /beds/availability`
- `GET /admissions`

## Forecasts and models

- `GET /forecast/history?days=90` — historical daily operations
- `GET /forecast/admissions`
- `POST /forecast/admissions` — reads saved output from the requested ISO date (`as_of_date`)
- `GET /forecast/occupancy`
- `POST /forecast/occupancy` — reads saved output from the requested ISO date (`as_of_date`)
- `GET /forecast/icu` — ICU admissions flow, not ICU occupied-bed census
- `GET /los/predictions?limit=500`
- `POST /los/predict` — filters saved LOS predictions by `patient_id` and/or `department`; at least one selector is required, and both supplied selectors must match. Returns up to 10 matching rows plus the total `prediction_count`; it does not create a new prediction.
- `GET /no-show/predictions?limit=500`
- `GET /models/metrics`

## Dashboard summaries

- `GET /dashboard/overview`
- `GET /dashboard/beds`
- `GET /dashboard/staff`
- `GET /dashboard/appointments`
- `GET /dashboard/optimization`

## Optimization, simulation, and audit

- `POST /optimization/run`
- `GET /optimization/runs`
- `GET /optimization/runs/{run_id}`
- `GET /optimization/runs/{run_id}/allocations`
- `GET /recommendations`
- `POST /recommendations/generate` — derives a recommendation from saved resource-gap output
- `GET /alerts`
- `POST /alerts` — validates and persists an alert
- `POST /simulation/run` — reruns the OR-Tools CP-SAT bed allocator against adjusted scenario inputs
- `GET /simulation/runs`
- `GET /simulation/{simulation_id}`
- `POST /assistant/ask` — explanations grounded in saved outputs; supported what-if questions run the OR-Tools simulation before explanation; reports `assistant_provider` and `llm_configured`
- `GET /audit/runs`
- `GET /audit/assistant?limit=100` — recent assistant questions, provider/model, status, data sources, latency, and sanitized errors; API keys are not stored

Simulation inputs are bounded by the API schema. Runs are recorded in the
configured SQL database. OpenAI is the primary assistant provider and requires
a backend-only `OPENAI_API_KEY`; deterministic demo mode and optional local
Ollama are alternatives. Provider details and data handling are described in
[ai_assistant.md](./ai_assistant.md). The assistant explicitly distinguishes
the ICU admissions forecast from an ICU census forecast.
Forecast and LOS request schemas reject unrecognized fields so unsupported
filters or scenario parameters are not silently ignored.
