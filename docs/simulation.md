# What-if simulator

`POST /simulation/run` runs `backend/app/services/simulation_service.py`
(past runs: `GET /simulation/runs`, `GET /simulation/{simulation_id}`). The simulator **re-solves the
OR-Tools CP-SAT bed allocator and the emergency/ICU capacity step** on adjusted
copies of the latest saved run's inputs, then compares the result with that
run's saved optimized KPIs.

It does **not** retrain or rerun any forecast, rerun staff scheduling, or
change the saved pipeline outputs.

## Inputs and how they are applied

| Input | Applied as |
|---|---|
| Admissions multiplier `m` | For every department and day, `(m - 1) × predicted admissions × mean predicted LOS` extra bed demand is added to the saved projected occupied beds (floored at 0). Isolation bed demand is multiplied by `m`. This is a flat approximation: it does not model when the extra patients arrive or leave. |
| ICU bed delta `n` | `n < 0`: that many *available* ICU beds are set to Maintenance; the request is rejected if fewer are available. `n > 0`: scenario-only ICU beds are added. |
| Staff availability delta `y` | Assigned staff in the saved coverage table is scaled by `1 + y` and truncated to whole people; unfilled positions are recomputed. Shifts are not rescheduled. |

The bed inventory includes scenario-only surge beds sized from the saved
(unadjusted) forecast, the same rule the pipeline uses, so a scenario starts
from the baseline's capacity assumptions; extra demand from the multiplier is
not met by extra surge beds.

## Outputs

Baseline and scenario values for allocated and overflow bed-days (split into
staffing-driven and bed-constraint-driven), unmet isolation bed-days, staffing,
physical-capacity and emergency/ICU-reserve violations, ICU overflow, peak
projected occupied beds and unfilled staff positions. A rule-based
recommendation states which measured quantity got worse. Each run, with its
assumptions and metrics, is stored in the configured SQL database
(`SimulationRecord`, `SimulationResult`).

## Limitations

Results are synthetic stress tests on one saved scenario. They are not repeated
across seeds, have no uncertainty estimate, and are not forecasts or live
operational orders. Repeated-seed experiments with confidence intervals are
planned (see the project status table in the README).
