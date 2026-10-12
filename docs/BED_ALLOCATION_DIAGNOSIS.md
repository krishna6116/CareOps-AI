# Bed allocation: diagnosis of the 146-request study

## What was reported

The standalone study in `optimization/bed_allocation/bed_allocation_optimizer.py`
produced runs in which 146 bed requests on one planning day were mostly or
entirely unassigned. An earlier run was reported to assign 0 of 146, with 46
compatible patient-bed pairs and 109 requests with no compatible bed; its
outputs are no longer in the repository. The outputs still
in the repository (`optimization/bed_allocation/outputs/`, planning date
2026-09-23) assign 33 of 146, with 763 compatible pairs and 40 requests with
no compatible bed. Every unassigned request has the same generic reason: "No
safe compatible bed was available under the optimization constraints."

## Root cause

The problem was in how the study built its requests, not in the solver. CP-SAT
reported `OPTIMAL` and its independent validation passed. The low assignment
rate was built into the inputs:

| Finding | Evidence |
| --- | --- |
| The requests were not requests. The script takes admissions dated before the planning date, keeps the most recent 30 days, and samples 146 of them with `random_state=42` (`bed_allocation_optimizer.py`, `REQUEST_LIMIT = 146`). | The saved requests have admission dates from 2026-07-24 to 2026-08-23: 31 to 61 days **before** the 2026-09-23 planning date. In the data those patients had already been admitted, and most had been discharged. |
| A month of demand was put on one day. | The synthetic hospital receives about 11.5 admissions a day (mean 11.54, 1,009 days). 146 requests on one day is about 12.7 days of arrivals. |
| Capacity capped the result, whatever the solver did. | 36 beds were free on the planning date, so at most 36 of 146 could ever be assigned. 41 requests needed ICU when 2 ICU beds were free. |
| The reasons were not diagnostic. | One message for every unassigned request; no per-request reason codes. |
| The outputs are not reproducible from the current data. | None of the 146 saved `(patient_id, admission_date)` pairs exist in the current `data/synthetic/admissions.csv`. The outputs come from an earlier dataset version. |

A second, separate problem appears when the shift roster is used as a hard
staffing limit. The synthetic roster (`data/synthetic/shifts.csv`) was
generated independently of the ward census. General Medicine averages about
56 inpatients against a rostered nurse capacity of about 2 beds (least-staffed
shift × 4 beds per nurse). Some wards have no nurse at all on some shifts, for
example General Medicine Morning on 2026-10-05. A hard roster limit therefore
rejects almost every admission because of this data artifact: 55 of 315
assigned in the replay below.

## Repair

`optimization/bed_assignment.py` replaces the study and runs as a pipeline
stage after `resource_optimization.py`:

- **Requests** are the actual synthetic arrivals on each planning day. Their
  recorded beds are ignored.
- **Census** is rebuilt from the admission history for each day: a bed is
  occupied when `admission_date < day < discharge_date`. Beds in maintenance
  in the inventory snapshot are out of service on every day.
- **Policies.** The baseline is arrival-order first fit into the home ward.
  The optimizer is CP-SAT with severity-weighted assignment and an
  off-service penalty. Both use the same compatibility, reserve and staffing
  rules.
- **Reasons.** Every unassigned request gets the first rule that blocked it:
  no compatible bed, all compatible beds occupied, protected reserve, staffing
  limit, or beds taken by other requests that day. These are checked against
  the beds actually left after that policy's assignments. If a request is
  unassigned while a usable bed remains, the run raises an error.
- **Staffing** is reported as `staffing_exceedances` by default. The
  `rostered_staffing_hard` sensitivity variant enforces it.

## Result

Results come from `run_portfolio.py --as-of-date 2026-10-05`, a 28-day replay
from 2026-09-08 to 2026-10-05. They are in
`optimization/resource_planning/outputs/bed_assignment_*.csv` and `bed_assignment_summary.json`.

| Policy | Requests | Assigned | Off-service | Compatibility violations | Main unassigned reasons |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline (home-ward first fit) | 315 | 288 | 0 | 0 | 19 isolation patients with no isolation room in their home ward, 5 protected reserve, 3 beds taken |
| Optimized (CP-SAT) | 315 | 311 | 22 | 0 | 2 protected reserve, 1 no compatible bed, 1 beds taken |

All 28 daily CP-SAT models solved to `OPTIMAL`. The gain comes from placing
patients off-service in compatible general-ward or isolation beds, and from
giving scarce beds to higher-severity requests (high-severity assignment rate
92.4% → 99.2%). It does not come from relaxing any rule. The
sensitivity table (`bed_assignment_sensitivity.csv`) shows the effect of the
off-service penalty (0, 50, 900), of the reserve (0, 2, 4 beds) and of hard
rostered staffing.

These are replays on synthetic data. They show that the allocator behaves
correctly. They are not evidence of improvement in a real hospital.

`tests/test_bed_assignment.py` covers census reconstruction, ICU, isolation,
pediatric and maintenance compatibility, reserve protection, severity
priority, hard staffing, empty days, reason codes and reproducibility.
