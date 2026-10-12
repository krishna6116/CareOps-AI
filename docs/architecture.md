# CareOps-AI Architecture

CareOps-AI has two halves: a **batch pipeline** that generates synthetic data,
trains and backtests forecasts, and runs the OR-Tools planner, writing files;
and a **serving layer** (FastAPI + React) that reads those files. The API does
not train models or generate forecasts on request.

```mermaid
flowchart TB
  subgraph Batch["Batch pipeline: run_portfolio.py"]
    G[data/synthetic/generate_data.py] --> V[Source integrity checks]
    V --> F1[Admissions] & F2[Length of stay] & F3[Occupancy] & F4[ICU admissions] & F5[No-show]
    F1 & F2 & F3 & F4 --> P[Bed availability projection]
    P & F5 --> O[resource_optimization.py<br/>CP-SAT beds, staff, appointments, emergency]
    V --> RA[bed_assignment.py<br/>request-level replay]
    O & RA --> Q[reliability_scorecard.py]
    Q --> M[portfolio_run_manifest.json<br/>docs/MODEL_METRICS.md]
    V -. --exploratory .-> X[cleaning, EDA, feature tables]
  end
  subgraph Serve["Serving layer"]
    API[FastAPI backend/app] --> UI[React dashboard frontend/]
    API --> DB[(SQLite or PostgreSQL:<br/>alerts, simulation runs,<br/>assistant audit)]
    API --> SIM[What-if simulator<br/>reruns the CP-SAT allocator]
    API --> AI[Assistant: grounded answers<br/>+ optional LLM rephrasing]
  end
  M --> API
  O --> API
  RA --> API
  M -. --postgres .-> PG[(PostgreSQL careops schema)]
```

## Components

| Component | Where | What it does |
|---|---|---|
| Synthetic generator | `data/synthetic/generate_data.py` | Seeded patients, admissions, beds, staff, shifts, appointments and daily operations up to the scenario cutoff. The hospital opens empty on 2024-01-01; the first 21 days are warm-up and are excluded from modelling (`ml/scenario_config.py`). |
| Forecasts | `ml/` | Admissions, occupancy, ICU admissions, LOS, no-show. Admissions and occupancy are chosen by a 30-day rolling-origin backtest against naive baselines (`ml/forecast_evaluation.py`); published counts are whole numbers (`ml/count_outputs.py`). |
| Planner | `optimization/resource_optimization.py` | Bed projection, CP-SAT department/day bed allocation, staff scheduling, appointment reassignment, emergency/ICU capacity, scenario surge capacity, input audit. |
| Request-level allocation | `optimization/bed_assignment.py` | Replays each day's actual arrivals against the census rebuilt from history. Compares an arrival-order home-ward baseline with CP-SAT under the same compatibility, reserve and staffing rules. Every unassigned request gets a reason code. Writes `bed_assignment_*` outputs and a sensitivity table. |
| Scorecard and manifest | `reliability_scorecard.py`, `run_portfolio.py` | Section scores from evidence, capacity gates, run status. The run exits nonzero if any gate fails. |
| API | `backend/app/` | Serves saved CSV/JSON outputs, with pagination where results are large. Persists alerts, simulation runs and assistant audit records through SQLAlchemy (SQLite unless `CAREOPS_DATABASE_URL` is set; schema via Alembic). `/health`, `/ready`, `/meta`. |
| What-if simulator | `backend/app/services/simulation_service.py` | Re-solves the bed allocator on adjusted saved inputs; see [simulation.md](simulation.md). |
| Assistant | `backend/app/services/assistant_facts.py`, `ai_assistant.py` | Routes a question to an intent and computes the answer from saved outputs, with typed evidence and sources. Optional OpenAI/Ollama only rephrase; numbers are checked, with a labelled fallback ([ai_assistant.md](ai_assistant.md)). |
| Dashboard | `frontend/` | React + TypeScript pages over the API. |
| PostgreSQL handoff | `optimization/postgresql_integration.py` | Optional: `run_portfolio.py --postgres` loads one run into the `careops` schema in a single, idempotent transaction ([database.md](database.md)). |

## Limitations

All data is synthetic. This is an operations decision-support prototype, not a
clinical system, and none of its outputs are live staffing or placement orders.
