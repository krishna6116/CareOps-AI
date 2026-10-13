"""Finalize a manifest from already validated portfolio artifacts.

This recovery command is useful after a long training run is interrupted after
the optimizer has written its outputs. It never retrains models or changes
source data; it reruns the same structural and capacity gates used by the
normal portfolio runner before writing the manifest.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from run_portfolio import (
    OUTPUT,
    SYNTHETIC_DATA,
    validate_appointment_schedule,
    validate_forecast_artifacts,
    validate_source_data,
)


def main() -> None:
    as_of = date.fromisoformat("2026-09-30")
    row_counts = validate_source_data(as_of, SYNTHETIC_DATA)
    forecast_windows = validate_forecast_artifacts(as_of)

    audit = pd.read_csv(OUTPUT / "data_quality_audit.csv")
    failures = audit[audit["status"].eq("FAIL")]
    if not failures.empty:
        raise RuntimeError(
            "Resource input audit failed: "
            + ", ".join(failures["check"].astype(str))
        )

    schedule = pd.read_csv(OUTPUT / "optimized_appointment_schedule.csv")
    appointment_check = validate_appointment_schedule(schedule)
    allocation = pd.read_csv(OUTPUT / "bed_allocation_kpis.csv")
    optimized = allocation[allocation["scenario"].eq("optimized")]
    coverage = pd.read_csv(OUTPUT / "staff_coverage_kpis.csv")
    emergency = pd.read_csv(OUTPUT / "emergency_capacity_optimization.csv")
    gates = {
        "overflow_bed_days": int(optimized["overflow_bed_days"].sum()),
        "unfilled_staff_positions": int(coverage["unfilled_staff"].sum()),
        "icu_overflow_bed_days": int(emergency["icu_overflow"].sum()),
        "emergency_overflow_bed_days": int(emergency["emergency_overflow"].sum()),
        "unexplained_unassigned_appointments": appointment_check["unexplained_unassigned"],
    }
    if any(gates.values()):
        raise RuntimeError(f"Portfolio capacity gates failed: {gates}")

    capacity = pd.read_csv(OUTPUT / "capacity_expansion_plan.csv")
    beds = capacity[capacity["resource_type"].eq("temporary_surge_beds")]
    staff = capacity[capacity["resource_type"].eq("synthetic_surge_staff")]
    scorecard = json.loads(
        (OUTPUT / "reliability_scorecard.json").read_text(encoding="utf-8")
    )
    if not scorecard["all_sections_at_or_above_target"]:
        raise RuntimeError("Reliability scorecard remains below its 4/5 target.")

    manifest = {
        "project": "CareOps-AI",
        "data_scope": "fully_synthetic_portfolio_scenario",
        "scenario_as_of_date": as_of.isoformat(),
        "forecast_horizon_days": 30,
        "forecast_windows": forecast_windows,
        "synthetic_source_row_counts": row_counts,
        "source_integrity_checks_passed": True,
        "audit_failures": 0,
        "audit_warnings": audit.loc[
            audit["status"].eq("WARN"), "check"
        ].astype(str).tolist(),
        "optimized_capacity_gates": gates,
        "optimized_appointment_count": appointment_check["scheduled"],
        "synthetic_capacity_expansion": {
            "scenario_only": True,
            "temporary_surge_beds": int(beds["added_quantity"].sum()),
            "surge_staff_pool_headcount": int(staff["added_quantity"].sum()),
            "scheduled_surge_hours": float(staff["scheduled_hours"].sum()),
            "illustrative_expansion_cost_usd": float(
                capacity["planned_cost_usd"].sum()
            ),
            "costs_are_scenario_assumptions_not_market_quotes": True,
        },
        "reliability_scorecard": {
            "path": str(OUTPUT / "reliability_scorecard.json"),
            "minimum_target": scorecard["minimum_target"],
            "all_sections_at_or_above_target": True,
        },
        "not_for_clinical_or_live_operational_use": True,
    }
    (OUTPUT / "portfolio_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(OUTPUT / "portfolio_run_manifest.json")


if __name__ == "__main__":
    main()
