# CareOps-AI project audit and scorecard

Audit date: 2026-10-12. Scenario: `run_portfolio.py --as-of-date 2026-10-05`.
Scores use the project brief's rubric: 1 missing/broken, 2 partial/unreliable,
3 functional with significant gaps, 4 strong with minor gaps, 5 thoroughly
validated and polished. A score reflects tested evidence, not the existence of
a file or a screen.

## Starting point

- Local `master` was 15 commits ahead of `origin/master`, with uncommitted
  evaluation work (rolling backtests for LOS, ICU and no-show; a dataset
  API).
- 126 tests passed. 2 errored because the occupancy outputs had not been
  regenerated after its code changed.
- Rerunning the seeded generator reproduced every committed source CSV byte for
  byte. The current data has 50,000 patients, 11,640 admissions (the older
  120,000 figure belongs to `data/processed`, an exploratory copy), 230 beds,
  300 staff, 120,000 appointments, 166,485 shifts and 9,081 daily-operation
  rows.
- `ModuleNotFoundError: backend` no longer reproduced: `pytest.ini` sets
  `pythonpath = .`.

## Issues found and fixed

| Issue | Finding | Fix | Evidence |
|---|---|---|---|
| 146 bed requests mostly unassigned | The standalone study sampled admissions 31–61 days old and put them on one day. Only 36 beds were free, so at most 36 of 146 could ever be placed. The saved outputs come from an older dataset. | `optimization/bed_assignment.py` replays actual daily arrivals against the census rebuilt for each day. It compares a baseline with CP-SAT and gives every unplaced request a reason code. | [BED_ALLOCATION_DIAGNOSIS.md](BED_ALLOCATION_DIAGNOSIS.md); `tests/test_bed_assignment.py` (12) |
| Roster staffing artifact | The synthetic shift roster is not tied to census. General Medicine averages ~56 inpatients against ~2 rostered nurse beds, so a hard limit places 55 of 315 requests. | Staffing is reported as exceedances by default; the hard variant is in the sensitivity table. | `bed_assignment_sensitivity.csv` |
| Assistant needed an API key | The default provider `openai` returned 503 without a key. The rules-based fallback misrouted most of the brief's example questions, for example by answering "beds available in 7 days" with one department's first-day row. | Grounded intent engine with typed evidence. The deterministic provider is the default. LLMs only rephrase, with number checks and a labelled fallback. | `assistant_facts.py`; 13 new API tests |
| Inconsistent coverage figures | The assistant mixed two interval methods (occupancy 80% coverage 73.8–87.5%) while the README reported the published method (74–83%). | Filtered to the published method in both the assistant and the API. | `/models/evaluation`, assistant answer |
| Wrong interval column | The dashboard and assistant read `upper_80`; the occupancy file uses `occupied_beds_upper_80`, so the interval was silently missing. | Corrected. | Forecasting chart; assistant text |
| Decimal counts | The simulator returned `projected_admissions` with 2 decimals and the peak as a float; dashboard counts used 1 decimal. | Integer API fields; `formatCount` everywhere counts are shown. | `test_simulation_count_metrics_are_whole_numbers`; Vitest |
| Mislabelled peak | The simulator's "peak projected occupied beds" was the largest single department-day row, not a hospital-wide peak. | Departments summed per day, then the maximum taken. | API test |
| 14 MB payloads | The dashboard downloaded all 23,470 no-show predictions twice to count risk bands. | `/no-show/summary` counts bands on the server; lists are capped at 500. | Live smoke test |
| Deprecated APIs | `datetime.utcnow` (50 warnings) and `on_event` in FastAPI. | Timezone-aware helper and lifespan handler. | Warnings 74 → 1 (third-party) |
| CI gaps | Node 20, but Vitest 5 and jsdom need Node 22.12+; no PostgreSQL, Docker or frontend tests in CI. | Five-job workflow (see deployment.md). | `.github/workflows/ci.yml` |

## Requirements traceability

| Brief phase | Implementation | Tests | API / page | Status |
|---|---|---|---|---|
| 1 Audit | this document | — | — | Done |
| 2 Data engineering and quality | `data/synthetic/generate_data.py`, `run_portfolio.validate_source_data`, `resource_validation.audit_resource_inputs`, `backend/app/services/data_service.py`, `data/build_data_dictionary.py` | `test_portfolio_pipeline.py`, `test_resource_validation.py`, `test_data_service.py` | `/datasets`, `/validation/datasets`; Alerts | Done; no automated schema-drift alerting between runs |
| 3 EDA and features | `data/eda_complete.py`, `data/feature_engineering.py` (exploratory); model-specific lag and rolling features | `test_leakage_audit.py` | — | Implemented; EDA outputs not re-reviewed in this audit |
| 4 Admissions | `ml/admission_forecasting.py`, `forecast_evaluation.py` | `test_forecast_evaluation.py`, `test_count_outputs.py` | `/forecast/admissions`; Forecasting | Done (baseline published) |
| 4 Occupancy | `ml/occupancy_forecasting.py` | same | `/forecast/occupancy` | Done |
| 4 LOS | `ml/los_prediction/los_prediction.py`, `ml/los_evaluation.py` | `test_forecast_evaluation.py` | `/los/predictions` | Done |
| 4 ICU | `ml/icu_demand_forecasting/` | same | `/forecast/icu` | Done; admissions, not census |
| 4 No-show | `ml/no_show_prediction/`, `ml/classification_evaluation.py` | `test_classification_evaluation.py` | `/no-show/summary` | Done; no fairness audit |
| 4 Registry and persistence | model metadata JSON per model; `model_status` in published forecasts; manifest | `test_generated_docs.py` | `/models/evaluation`, `/models/metrics` | Implemented as files plus manifest; no separate registry service |
| 5 Bed availability | `resource_optimization.build_bed_forecast` | `test_resource_optimization.py` | `/beds/availability`; Bed management | Done |
| 6 Bed allocation | `resource_optimization.optimize_bed_allocations`, `optimization/bed_assignment.py` | `test_bed_assignment.py`, `test_resource_optimization.py` | `/optimization/bed-assignments*` | Done |
| 7 Staff scheduling | `resource_optimization.optimize_staff_schedule` | `test_resource_optimization.py` | `/dashboard/staff` | Partial: no explicit overtime model |
| 8 Appointments | `resource_optimization.optimize_appointments` | same | `/dashboard/appointments` | Partial: no-show risk not in the objective |
| 9 Emergency capacity | `resource_optimization.optimize_emergency_capacity` | same | `/dashboard/optimization` | Done |
| 10 End-to-end backend | `run_portfolio.py` (9 stages and gates), FastAPI services | `test_portfolio_pipeline.py`, API tests, PostgreSQL handoff test | all | Done |
| 11 PostgreSQL | Alembic, `database.py` pool, `postgresql_integration.py` | `test_postgres_integration.py` (6, on PostgreSQL 17.6) | `/health`, `/ready` | Done |
| 12 FastAPI | `backend/app` | `test_api.py` (40) | `/docs` | Done; no authentication |
| 13 Dashboard | `frontend/src` | 20 Vitest tests | all pages | Done; not checked in a real browser this session |
| 14 Explanations and recommendations | `assistant_facts.py`, `dashboard_service.get_recommendations` | API tests | `/assistant/ask`, `/recommendations` | Done |
| 15 What-if | `simulation_service.py` | API tests, PostgreSQL restart test | `/simulation/*`; Simulator | Done; forecasts scaled, not retrained |
| 16 Business metrics | KPI CSVs, [validation.md](validation.md) | optimization tests | Optimization, Bed management | Done |
| 17 Testing | 154 Python + 6 PostgreSQL + 20 frontend | CI | — | Done; no browser end-to-end test |
| 18 Build and runtime | pipeline, API smoke, frontend build | CI | — | Done |
| 19 Docker and CI/CD | `Dockerfile`, `docker-compose.yml`, `ci.yml` | CI docker-compose job | — | Done; all 5 CI jobs pass |
| 20 Documentation | README, `docs/` | `test_generated_docs.py` | — | Done |
| 21 Security | `.gitignore`, `.dockerignore`, `.env.example`, non-root image | secret scan of all commits (clean) | — | Done |
| 22 Publication | `origin` = github.com/krishna6116/CareOps-AI | — | — | Done |

## Scorecard

| # | Category | Score | Evidence | Known limitations / remaining work |
|---|---|---|---|---|
| 1 | Synthetic-data quality | 4.0 | Seeded and byte-reproducible; 0 orphan foreign keys; no overlapping bed stays; LOS consistent with dates | Roster not tied to census; isolation labels added afterwards; arrivals close to Poisson by design |
| 2 | Data engineering | 4.0 | Source integrity gates, input audit, dataset catalog and validation API, generated data dictionary | No automated schema-drift detection between runs |
| 3 | EDA and feature engineering | 3.5 | Exploratory stages reproducible; leakage audit tests | EDA outputs not re-reviewed in this audit; features are model-specific |
| 4 | Admission forecasting | 4.5 | 24-origin rolling backtest, baselines, whole-number intervals with measured coverage; baseline published honestly | Little signal to learn in the synthetic data |
| 5 | Occupancy forecasting | 4.5 | Ridge beats all baselines (MAE 8.72 vs 11.88); intervals; alert-threshold scoring | 90%/95% thresholds never reached, so alerting there is untested |
| 6 | LOS prediction | 4.0 | Rolling 30-day windows vs department × severity baseline (2.62 vs 3.04) | Censoring is handled by training scope only; no per-department report in the UI |
| 7 | ICU demand | 4.0 | Rolling backtest; baseline published; census projected in planning | No direct census forecast |
| 8 | No-show prediction | 4.0 | Calibrated XGBoost vs baselines; Brier, PR-AUC, ECE; validation-chosen thresholds | No fairness audit; not used in scheduling |
| 9 | Bed availability | 4.0 | Integrated projection, never negative, ICU and surge handled | No department-level intervals |
| 10 | Bed allocation | 4.5 | Request-level replay, reason codes, sensitivity, 12 unit tests, diagnosis report | Daily batching; staffing reported rather than enforced (data artifact) |
| 11 | Staff scheduling | 3.5 | CP-SAT with availability, hours, rest and a workload penalty; 0 unfilled shifts with surge pool | No overtime model; no baseline-vs-optimized staffing comparison in the UI |
| 12 | Appointment optimization | 3.5 | Conflict-free reassignment (8 → 0 doctor conflicts) | Latest day only; no-show risk not used |
| 13 | Emergency capacity | 4.0 | ICU/emergency reserve and overflow per day; thresholds in code | Thresholds not configurable at runtime |
| 14 | PostgreSQL | 4.5 | Migrations with drift check; 6 integration tests on PostgreSQL 17.6; restart persistence shown with a real process | Read endpoints serve files rather than the database (documented) |
| 15 | FastAPI backend | 4.0 | Health, ready and meta; validation; pagination on large endpoints; 40 API tests | Many routes return untyped dicts; no authentication |
| 16 | React frontend | 4.0 | Every page reads the API; count formatting; 20 tests; production build | `App.tsx` is still large; not checked in a browser this session |
| 17 | AI assistant | 4.5 | Grounded intents, typed evidence, sources, labelled fallback, number checks; tests cover every example question | Keyword routing; unsupported phrasings get the overview answer |
| 18 | What-if simulation | 4.0 | Validated inputs; persisted; states what was rerun; hospital-wide integer KPIs | Forecasts scaled, not retrained; no repeated seeds |
| 19 | Explainability and auditability | 4.0 | Manifest, scorecard, reason codes, assistant audit, simulation history | Only the latest pipeline run is kept as a manifest |
| 20 | Automated testing | 4.0 | 154 + 6 + 20 tests across data, models, optimization, API, database and UI; all run in CI | No browser end-to-end test |
| 21 | Docker and deployment | 4.0 | CI `docker-compose` job (run 38275091237, commit 51313e7) builds the images, waits on all three healthchecks, calls the API, assistant and dashboard, and confirms simulation history survives an API restart on PostgreSQL. Non-root backend image. | Not run on a local Docker host (Docker Desktop not installed here); no public demo deployment |
| 22 | CI/CD | 4.5 | Five jobs all passed on commit 51313e7: Python tests + migration drift check, PostgreSQL 16 integration (fails if skipped), 25% pipeline smoke run, frontend type check/tests/build, Docker Compose stack | No deployment stage (no hosting account) |
| 23 | Documentation | 4.5 | README, generated metrics and data dictionary, diagnosis, troubleshooting, per-component docs | — |
| 24 | Security and hygiene | 4.0 | No secrets in any commit; `.env` ignored; non-root image; localhost-bound ports; redacted audit | No API authentication or rate limiting |
| 25 | GitHub publication | 4.5 | 24 commits fast-forwarded to `origin/master` (5b11c5c → 51313e7) with no force push; remote verified equal to local; full-history secret scan clean | — |

Categories below 4.5 are scored honestly against the evidence above. The
largest remaining gaps are the appointment and staff objectives, a browser
end-to-end test, and API authentication.
