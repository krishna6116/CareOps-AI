# What-if simulator

The simulator reruns the project's OR-Tools CP-SAT bed allocator using the
latest saved forecast, bed inventory, and staff coverage inputs. It compares
the solver result with the saved optimized baseline and persists scenario
assumptions and calculated metrics through the configured SQL database.

Inputs include an admissions multiplier, ICU bed capacity delta, and staff
availability delta. Extra admissions are estimated using each department's
saved forecasted admissions and mean predicted length of stay. ICU capacity
changes are applied to available ICU inventory; a request to remove more beds
than are available is rejected. Staff availability changes are applied to the
saved shift coverage before the solver runs.

Results include calculated overflow, allocation, isolation demand, staffing
gaps, capacity/reserve violations, and ICU overflow where available. The
recommendation is based on those calculated differences. This is a synthetic
scenario stress test, not a substitute for a fresh clinical forecast or a live
operational plan.
