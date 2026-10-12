# Validation and business metrics

Every reported number comes from a pipeline run. These are the sources:

| Result type | Where | Denominator / definition |
|---|---|---|
| Forecast accuracy | `docs/MODEL_METRICS.md`, `/models/evaluation` | MAE, RMSE, sMAPE, MASE and bias over 30-day rolling-origin backtests; each model is compared with naive baselines on the same origins |
| Interval coverage | same | Share of actuals inside the published whole-number 80% / 95% intervals, by horizon bucket |
| Classification (no-show) | same | Brier, log loss, ROC-AUC, PR-AUC, ECE; thresholds chosen on validation data |
| Department bed plan | `bed_allocation_kpis.csv` | Overflow bed-days and violations over the 30-day horizon; baseline and optimized share the forecast and capacity |
| Request-level allocation | `bed_assignment_kpis.csv` | Assignment rate = assigned ÷ all requests in the 28-day window. High-severity rate = assigned ÷ Critical and High requests. Free-bed fill rate = assigned ÷ in-service beds free at the start of each day, pooled |
| Staff coverage | `staff_coverage_kpis.csv` | Required vs assigned staff per date, department, shift and role |
| Appointments | `appointment_optimization_kpis.csv` | Conflicts, off-shift and unavailable-provider bookings, before vs after |
| What-if scenarios | `/simulation/runs` | Scenario vs the saved optimized baseline under stated assumptions |

## Rules

- Comparisons share the same data, rules and window. The baseline policy is
  stated with each result.
- If a learned model does not beat its baseline, the baseline is published and
  the README says so.
- Planning results show that plans are feasible and that rules are respected
  under synthetic assumptions. They are not improvements measured in a real
  hospital.
- Each run records its seeds, cutoff date, model choices, solver statuses and
  quality gates in `portfolio_run_manifest.json` and
  `reliability_scorecard.json`.

The project is not for live clinical or operational use.
