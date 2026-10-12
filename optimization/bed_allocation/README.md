# Superseded: standalone bed-allocation study

This folder holds an early standalone study. It is kept for reference and is
not part of the pipeline, the API or the dashboard.

Its requests were a random sample of admissions from 31 to 61 days before the
planning date, all placed on a single day. That design caps the assignment
rate whatever the solver does, and the saved outputs come from an earlier
dataset version. See [docs/BED_ALLOCATION_DIAGNOSIS.md](../../docs/BED_ALLOCATION_DIAGNOSIS.md).

Request-level allocation is now done by
[optimization/bed_assignment.py](../bed_assignment.py). It replays actual
arrivals against the census rebuilt for each day, compares a baseline policy
with CP-SAT, and gives a reason code for every unassigned request.
