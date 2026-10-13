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
| CI gaps | Node 20, but Vitest 5 and jsdom need Node 22.12+; no PostgreSQL, Docker or frontend tests in CI. | Six-job workflow, plus Pages (see deployment.md). | `.github/workflows/ci.yml` |
| Summed percentages | `/forecast/history` and `/operations` summed the per-department `occupancy` percentage (e.g. 542 on 2026-10-04), so the "historical occupied beds" chart sat far above the forecast. | Sum count columns only; occupied beds come from `bed_demand` (143, equal to the census). | `test_history_reports_occupied_beds_not_summed_percentages` |
| Overtime weighting | A first overtime model priced overtime below a surge shift, giving 2,211 overtime shifts (14,368 h) in 30 days. | Full overtime shift costed above a surge shift: 424 mostly partial overtime shifts (1,500 h). | `staff_schedule_comparison.csv`; staffing tests |
| Appointment solver | Scheduling 7 days instead of 1: CP-SAT returned UNKNOWN on the busiest day (143 appointments, ~28k optional intervals) because probing presolve used the whole time limit. | Greedy conflict-free hint and probing off: about 25 s per day, all 838 booked. | `test_last_n_days_are_each_scheduled_without_conflicts` |
| Browser-only defects | A browser showed a 404 on every page (no favicon), a half-width alerts table wrapping one word per line, and dates wrapping onto three lines. | Favicon; full-width alerts; long-text cells wrap, short cells do not. | Playwright fails on console errors; screenshots reviewed |
| Overstated claim | The README called admissions "Poisson noise around a stable level". EDA found a significant weekday effect (weekends about 15% lower) and weak monthly seasonality. | Wording corrected in the README, metrics report, dashboard and assistant. | `docs/EDA_FINDINGS.md` |

## Requirements traceability

| Brief phase | Implementation | Tests | API / page | Status |
|---|---|---|---|---|
| 1 Audit | this document | — | — | Done |
| 2 Data engineering and quality | `data/synthetic/generate_data.py`, `run_portfolio.validate_source_data`, `resource_validation.audit_resource_inputs`, `backend/app/services/data_service.py`, `data/build_data_dictionary.py` | `test_portfolio_pipeline.py`, `test_resource_validation.py`, `test_data_service.py` | `/datasets`, `/validation/datasets`; Alerts | Done; no automated schema-drift alerting between runs |
| 3 EDA and features | `data/eda_findings.py` → `docs/EDA_FINDINGS.md`; `data/eda_complete.py`, `data/feature_engineering.py` (exploratory) | `test_leakage_audit.py` | — | Done |
| 4 Admissions | `ml/admission_forecasting.py`, `forecast_evaluation.py` | `test_forecast_evaluation.py`, `test_count_outputs.py` | `/forecast/admissions`; Forecasting | Done (baseline published) |
| 4 Occupancy | `ml/occupancy_forecasting.py` | same | `/forecast/occupancy` | Done |
| 4 LOS | `ml/los_prediction/los_prediction.py`, `ml/los_evaluation.py` | `test_forecast_evaluation.py` | `/los/predictions` | Done |
| 4 ICU | `ml/icu_demand_forecasting/` | same | `/forecast/icu` | Done; admissions, not census |
| 4 No-show | `ml/no_show_prediction/`, `ml/classification_evaluation.py` | `test_classification_evaluation.py` | `/no-show/summary` | Done; no fairness audit |
| 4 Registry and persistence | model metadata JSON per model; `model_status` in published forecasts; manifest | `test_generated_docs.py` | `/models/evaluation`, `/models/metrics` | Implemented as files plus manifest; no separate registry service |
| 5 Bed availability | `resource_optimization.build_bed_forecast` | `test_resource_optimization.py` | `/beds/availability`; Bed management | Done |
| 6 Bed allocation | `resource_optimization.optimize_bed_allocations`, `optimization/bed_assignment.py` | `test_bed_assignment.py`, `test_resource_optimization.py` | `/optimization/bed-assignments*` | Done |
| 7 Staff scheduling | `resource_optimization.optimize_staff_schedule` (optimized and first-fit) | `test_resource_optimization.py` | `/dashboard/staff`; Staff page | Done |
| 8 Appointments | `resource_optimization.optimize_appointments` (7 days) | same | `/dashboard/appointments`; Appointments page | Done; risk used for outreach, never for booking |
| 9 Emergency capacity | `resource_optimization.optimize_emergency_capacity` | same | `/dashboard/optimization` | Done |
| 10 End-to-end backend | `run_portfolio.py` (9 stages and gates), FastAPI services | `test_portfolio_pipeline.py`, API tests, PostgreSQL handoff test | all | Done |
| 11 PostgreSQL | Alembic, `database.py` pool, `postgresql_integration.py` | `test_postgres_integration.py` (6, on PostgreSQL 17.6) | `/health`, `/ready` | Done |
| 12 FastAPI | `backend/app`, `security.py` | `test_api.py` (46) | `/docs` | Done |
| 13 Dashboard | `frontend/src`, `frontend/e2e` | 20 Vitest + 16 Playwright tests | all pages | Done |
| 14 Explanations and recommendations | `assistant_facts.py`, `dashboard_service.get_recommendations` | API tests | `/assistant/ask`, `/recommendations` | Done |
| 15 What-if | `simulation_service.py` | API tests, PostgreSQL restart test | `/simulation/*`; Simulator | Done; forecasts scaled, not retrained |
| 16 Business metrics | KPI CSVs, [validation.md](validation.md) | optimization tests | Optimization, Bed management | Done |
| 17 Testing | 167 Python + 6 PostgreSQL + 20 Vitest + 16 browser (also run against the deployed demo) | CI | — | Done |
| 18 Build and runtime | pipeline, API smoke, frontend build | CI | — | Done |
| 19 Docker, CI/CD, demo | `Dockerfile`, `docker-compose.yml`, `ci.yml`, `pages.yml`, `deploy/export_demo_snapshot.py` | CI docker-compose and browser-e2e jobs | public demo | Done |
| 20 Documentation | README, `docs/` | `test_generated_docs.py` | — | Done |
| 21 Security | `.gitignore`, `.dockerignore`, `.env.example`, non-root image | secret scan of all commits (clean) | — | Done |
| 22 Publication | `origin` = github.com/krishna6116/CareOps-AI | — | — | Done |

## Scorecard

| # | Category | Score | Evidence | Known limitations / remaining work |
|---|---|---|---|---|
| 1 | Synthetic-data quality | 4.0 | Seeded and byte-reproducible; 0 orphan foreign keys; no overlapping bed stays; LOS consistent with dates | Roster not tied to census; isolation labels added afterwards; arrivals close to Poisson by design |
| 2 | Data engineering | 4.0 | Source integrity gates, input audit, dataset catalog and validation API, generated data dictionary | No automated schema-drift detection between runs |
| 3 | EDA and feature engineering | 4.5 | `docs/EDA_FINDINGS.md` is generated from the data with statistical tests (dispersion 1.05; weekday Kruskal-Wallis p < 0.001; monthly p = 0.01; reminder chi-square p ≈ 2e-91), and each finding is tied to a modelling decision; leakage audit tests | Feature tables are model-specific rather than a shared feature store |
| 4 | Admission forecasting | 4.5 | 24-origin rolling backtest, baselines, whole-number intervals with measured coverage; baseline published honestly | Little signal to learn in the synthetic data |
| 5 | Occupancy forecasting | 4.5 | Ridge beats all baselines (MAE 8.72 vs 11.88); intervals; alert-threshold scoring | 90%/95% thresholds never reached, so alerting there is untested |
| 6 | LOS prediction | 4.0 | Rolling 30-day windows vs department × severity baseline (2.62 vs 3.04) | Censoring is handled by training scope only; no per-department report in the UI |
| 7 | ICU demand | 4.0 | Rolling backtest; baseline published; census projected in planning | No direct census forecast |
| 8 | No-show prediction | 4.0 | Calibrated XGBoost vs baselines; Brier, PR-AUC, ECE; validation-chosen thresholds | No fairness audit; not used in scheduling |
| 9 | Bed availability | 4.0 | Integrated projection, never negative, ICU and surge handled | No department-level intervals |
| 10 | Bed allocation | 4.5 | Request-level replay, reason codes, sensitivity, 12 unit tests, diagnosis report | Daily batching; staffing reported rather than enforced (data artifact) |
| 11 | Staff scheduling | 4.5 | Costed overtime (cap one extra shift a week) and a first-fit baseline on the same rules: unfilled 27 → 0, surge shifts 2,512 → 2,365, workload std 44.1 → 31.0 h, preference match 52.5% → 92.2%; a rejected weighting is documented; 6 new tests; independent validator checks the overtime cap | Overtime and surge costs are illustrative weights; the roster is synthetic |
| 12 | Appointment optimization | 4.5 | 7 clinic days: 838/838 booked; conflicts 77 → 0; off-shift bookings 254 → 0; utilization with and without expected no-shows; risk used only for outreach (217) and idle-time estimates, with a test showing bookings are identical at low and high risk | Fixed 15-minute slots; no explicit overbooking decision (left to staff) |
| 13 | Emergency capacity | 4.0 | ICU/emergency reserve and overflow per day; thresholds in code | Thresholds not configurable at runtime |
| 14 | PostgreSQL | 4.5 | Migrations with drift check; 6 integration tests on PostgreSQL 17.6; restart persistence shown with a real process | Read endpoints serve files rather than the database (documented) |
| 15 | FastAPI backend | 4.5 | Typed response models on the main routes; API key on admin endpoints (fails closed outside development); per-client rate limits on solver endpoints; pagination; 46 API tests | Row-level tables stay untyped dicts; no user accounts or roles |
| 16 | React frontend | 4.5 | 16 Playwright tests pass in the installed Google Chrome against the live API and the static demo; screenshots reviewed and the layout defects fixed; 20 Vitest tests; staffing and appointment comparisons | `App.tsx` still holds most pages |
| 17 | AI assistant | 4.5 | Grounded intents, typed evidence, sources, labelled fallback, number checks; tests cover every example question | Keyword routing; unsupported phrasings get the overview answer |
| 18 | What-if simulation | 4.0 | Validated inputs; persisted; states what was rerun; hospital-wide integer KPIs | Forecasts scaled, not retrained; no repeated seeds |
| 19 | Explainability and auditability | 4.0 | Manifest, scorecard, reason codes, assistant audit, simulation history | Only the latest pipeline run is kept as a manifest |
| 20 | Automated testing | 4.5 | 166 Python, 6 PostgreSQL, 20 Vitest and 16 browser tests; all run in CI | No load or performance tests |
| 21 | Docker and deployment | 4.5 | CI `docker-compose` job (run 38444586244, commit 19dcf72) builds the images, waits on all three healthchecks, calls the API, assistant and dashboard, and confirms simulation history survives an API restart on PostgreSQL. Non-root backend image. Public demo deployed by Pages run 38444586274 at https://krishna6116.github.io/CareOps-AI/; all 16 Playwright tests pass against the deployed site in Google Chrome | Docker not run on a local host (Docker Desktop not installed here); the demo is a static snapshot, not a live API |
| 22 | CI/CD | 4.5 | All six jobs passed on commit 19dcf72 (run 38444586244): Python tests and migration drift check, PostgreSQL 16 integration (fails if skipped), 25% pipeline smoke run, frontend type check/tests/build, browser end-to-end in Chromium (live API and demo build), Docker Compose stack. The Pages workflow deploys the demo on every push. CI caught two real defects in this round (the 7-day appointment gate, the readiness gate); both were fixed | GitHub warns that Node 20 actions and ubuntu-latest will change versions |
| 23 | Documentation | 4.5 | README, generated metrics and data dictionary, diagnosis, troubleshooting, per-component docs | — |
| 24 | Security and hygiene | 4.5 | API key on admin endpoints, rate limits, no secrets in any commit, `.env` ignored, non-root image, localhost-bound ports, redacted assistant audit, pinned dependencies of at least two weeks' age | No user accounts; the limiter is per process |
| 25 | GitHub publication | 4.5 | All work fast-forwarded to `origin/master` with no force push (5b11c5c → 51313e7 → ee639e6 → 19dcf72 and this commit); remote verified equal to local; full-history secret scan clean; public demo live | — |

Categories below 4.5 are scored honestly against the evidence above. The
remaining gaps:

- Synthetic data: the roster is not tied to census, and arrivals are close to
  Poisson.
- Data engineering: no schema-drift alerts between runs.
- LOS, ICU and no-show models: no prospective validation, no direct ICU census
  model, no fairness audit.
- Department-level availability: no intervals.
- Emergency thresholds: fixed in code.
- What-if: forecasts are scaled, not retrained.
- Run history: only the latest manifest is kept.
