# Resource optimization (Phases 4–5)

Run the complete synthetic portfolio pipeline from the project root:

```powershell
.\.venv\Scripts\python.exe run_portfolio.py
```

The pipeline regenerates source fixtures under `data/synthetic/`, retrains
admission, LOS, occupancy, ICU-admission, and no-show models in order, generates
resource support inputs, runs the planner, and verifies that all forecast
artifacts cover the same 30-day window. It leaves `data/processed/` unchanged.
The default scenario cutoff is today; `--as-of-date YYYY-MM-DD` makes a run
reproducible for a chosen date. Outputs are written to
`optimization/resource_planning/outputs/`. CP-SAT uses a fixed seed and one
search worker.

The generator keeps expected demand feasible against the configured 230-bed
inventory. Admission events are assigned compatible, non-overlapping beds;
daily operations are then aggregated from those same events, including actual
discharges, ICU admissions, census, and occupancy rates. The cutoff-date bed
snapshot is reconciled to active assignments. Stays not discharged by the
scenario cutoff are labeled `in_progress` and excluded from LOS model fitting
to avoid using outcomes that would not yet be observed. Appointments are
generated chronologically with matching departments/providers, and shifts are
checked against rolling weekly hours and night-to-morning rest constraints.

Before model training, the portfolio runner fails on broken identifiers,
foreign keys, invalid stay dates/LOS, bed assignment overlaps or mismatches,
provider mismatches, shift-hour/rest violations, or discrepancies among
admission events, daily operations, census, and the hospital snapshot. A
successful `portfolio_run_manifest.json` is only written after the end-to-end
forecast, audit, and zero-overflow scenario gates pass.

To run only the planner against the existing processed data and saved forecast
artifacts, use:

```powershell
.\.venv\Scripts\python.exe optimization\resource_optimization.py
```

Before optimization, the planner writes `data_quality_audit.csv` and scores
saved chronological test predictions into `forecast_backtest_summary.csv`.
Duplicate IDs/grains, invalid forecast values, impossible admission-stay
dates, or forecasts with inconsistent date horizons stop the run. Synthetic
provenance and denominator assumptions remain visible for review. To fail
closed when inputs are stale or operational
snapshots have no timestamp, set this PowerShell environment variable:

```powershell
$env:CAREOPS_REQUIRE_FRESH_DATA = "1"
.\.venv\Scripts\python.exe optimization\resource_optimization.py
Remove-Item Env:\CAREOPS_REQUIRE_FRESH_DATA
```

Synthetic scenario inputs are expected to fail this strict gate. It is a
production-readiness guard, not a test failure for the portfolio simulation.
The runner creates an explicit support-data bundle in
`data/synthetic/resource_support/`. Its dated snapshots are assumptions based
on the scenario cutoff date, and its seeded isolation flags/rates are
synthetic scenario labels, not clinical determinations. The ICU forecasting
script now emits a 30-day daily-admissions forecast. Daily ICU admissions are a
flow, not occupied ICU beds: they are kept as a separate signal and are never
compared directly with the physical ICU bed count. ICU bed allocation is based
on the occupancy-calibrated departmental census projection. Occupancy
denominators come from the generated bed inventory; the audit records any
physical-versus-operational capacity difference rather than assuming they match.

## Included planning work

- Bed availability forecast: combines current bed status/occupancy, department
  admission forecasts, department LOS predictions, expected discharges, and the
  hospital-wide occupancy forecast. Department-level admission/LOS projections
  are calibrated to the independent occupancy forecast; both calibrated and
  uncalibrated projections are retained for review. Synthetic, rate-derived
  isolation admissions are converted to a projected isolation-bed census using
  department LOS and are separately identified in the forecast output with
  their scenario rate and synthetic provenance.
- Bed allocation: compares department-first allocation with a CP-SAT plan that
  observes specialty, staffing, ICU/emergency reserve, and overflow/transfer
  constraints. `bed_allocation_kpis.csv` separates forecast occupancy,
  allocation as a fraction of physical beds, and allocation as a fraction of
  staffed capacity. It attributes overflow between staffing and dedicated
  bed/compatibility limits. Overflow bed-days per forecast admission is a
  workload indicator, not a patient waiting-time estimate. The existing
  patient-level allocator also enforces isolation when an `isolation_required`
  field is present. Historical admissions are joined to seeded synthetic
  isolation labels for a reproducible scenario; these labels do not establish
  actual patient isolation requirements.
- Staff scheduling: plans available clinical staff across the full forecast
  horizon using observed role/department/shift coverage, forecast workload,
  department/specialty, weekly hours, minimum rest, and shift preferences. For
  the portfolio scenario, nurse requirements are also derived from forecast
  occupied beds using the stated four-beds-per-nurse proxy. A labeled,
  scenario-only flex roster is sized to meet these positions while retaining
  hour and rest constraints. The planner stops if any positions remain unfilled.
  Base staff may work one extra shift per rolling 7 days as overtime. A full
  overtime shift (8 h x 500) is costed above a surge assignment (3,000), so
  overtime appears only where it is cheaper than calling in surge staff
  (mostly partial shifts). A first-fit baseline (roster order, base staff
  first, no overtime) runs on the same requirements and rules;
  `staff_schedule_comparison.csv` compares unfilled positions, surge and
  overtime use, workload spread and shift-preference match.
- Capacity expansion: forecast peak census determines explicit temporary surge
  bed counts by department, including ICU/emergency protected reserves and
  isolation-compatible capacity. The `capacity_expansion_plan.csv` file lists
  added beds, flex roster headcount, scheduled flex hours, and illustrative
  costs. Those resources are assumptions for the synthetic optimized scenario,
  not existing inventory, confirmed hiring, funding, or market-rate estimates.
  The constrained baseline remains in the KPI comparison. The optimized run
  fails closed if any bed, ICU, or emergency overflow remains.
- Appointment scheduling: re-slots each of the last 7 clinic days (one CP-SAT
  model per day) using available specialty-matched doctors, 15-minute slots,
  doctor non-overlap, duration, and priority. A greedy conflict-free schedule
  is supplied as a solution hint and presolve probing is off, so a busy day
  (~28k optional intervals) cannot exhaust the time limit with no solution.
  No-show risk never changes who is booked, their priority or their slot: it
  flags reminder outreach at the model's validation-chosen threshold and
  estimates expected idle minutes (risk x duration) for overbooking review.
  Missing scores use only prior-date department history, never same-day outcome
  labels. The synthetic provider roster and shifts are generated with
  department-level appointment coverage targets. An appointment may stay
  unbooked only for a recorded, provable reason: no doctor of its specialty
  is on shift in clinic hours that day (`no_specialty_doctor_on_shift`), or
  every on-shift slot is taken on a day CP-SAT proved OPTIMAL
  (`on_shift_capacity_full`). Any other unbooked appointment fails the run;
  the manifest records the unassigned count and reasons. The full 2026-10-05
  run books all 838; the smaller CI smoke scenario has days with no
  specialist rostered.
- Emergency capacity: compares projected ICU bed census and emergency demand
  with physical bed and staffed capacity, protects the ICU reserve, and can
  use unoccupied General Medicine beds for emergency overflow only within
  available ED staffing capacity. Staffing capacity is taken from the optimized
  daily coverage for each forecast date.
- Resource gaps: `resource_gap_analysis.csv` estimates per-department dedicated
  bed shortfalls and additional nurses needed on each shift under the stated
  nurse-to-bed scenario ratio. These are scenario gaps, not hiring orders or
  evidence that float staff can safely cover another specialty.
- Input quality and retrospective performance: `data_quality_audit.csv` records
  data age, snapshot timestamps, key uniqueness, forecast validity, and
  source-to-forecast alignment. `forecast_backtest_summary.csv` scores saved
  chronological holdout predictions using MAE/RMSE/WAPE and baseline
  comparisons. Admission model complexity is selected on a chronological
  validation tail within training only; its final holdout remains untouched.
  The admission seasonal baseline updates from observations as each holdout day
  advances, matching the one-step-ahead evaluation. No-show scoring includes
  Brier score against pre-holdout historical prevalence and
  ROC-AUC for no-show probabilities). The no-show score is 4/5 when calibrated
  Brier score beats that baseline and is at most 0.17; it reaches 4.5/5 only
  when ROC-AUC is also at least 0.70 on the same chronological holdout. These
  are stored test-set metrics; they do not establish prospective or 30-day
  recursive accuracy. No-show probabilities flag reminder outreach but do
  not affect appointment time changes. Inspect department-level rows and do
  not rely on a single hospital-wide average.

## Output files

- `bed_availability_forecast.csv`
- `bed_allocation_simulation.csv`
- `bed_allocation_kpis.csv`
- `optimized_staff_schedule.csv`
- `staff_coverage_kpis.csv`
- `staff_schedule_comparison.csv`
- `optimized_appointment_schedule.csv`
- `appointment_optimization_kpis.csv`
- `emergency_capacity_optimization.csv`
- `resource_gap_analysis.csv`
- `capacity_expansion_plan.csv`
- `data_quality_audit.csv`
- `forecast_backtest_summary.csv`
- `reliability_scorecard.json`, `reliability_scorecard.csv`, and `reliability_scorecard.txt`
- `reliability_details.csv` (generated datasets and forecast model/group scores)
- `resource_optimization_report.txt`
- `portfolio_run_manifest.json` (full pipeline runs)

The reliability scorecard reports an equal-weighted overall portfolio score
and scores by section. Dataset scores reflect structural validation in this
synthetic scenario; each forecast/group score is based on its retrospective
chronological holdout and applicable baseline or error threshold. Future
forecast values have no observed outcomes yet, so these scores are not
confidence intervals or guarantees. Review the evidence and limitations in
the detailed report alongside each score.

These outputs are planning simulations, not clinical recommendations. The
scenario uses a configurable proxy of four beds per scheduled nurse and adds
explicit synthetic surge resources to the optimized scenario; the baseline
continues to represent the original constrained inventory. Thus zero optimized
overflow means only that the modeled capacity expansion covers the modeled
forecast—not that real-world overflow has been resolved. Costs are illustrative
assumptions and are not market quotes. There are no patient-level queue records,
so the project does not estimate actual waiting time. Results are sensitive to
forecast quality, staffing and bed assumptions, specialty mapping, and source
data. Review assumptions and any real-world recommendations with operational
and clinical teams.
