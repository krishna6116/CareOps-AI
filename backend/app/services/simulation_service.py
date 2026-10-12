from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd

from backend.app.database import SessionLocal
from backend.app.models.careops import SimulationRecord, SimulationResult
from backend.app.services.dashboard_service import get_overview
from backend.app.services.pipeline_service import OUTPUT_DIR, read_manifest


def _apply_icu_capacity_delta(beds: pd.DataFrame, delta: int) -> pd.DataFrame:
    adjusted = beds.copy()
    department = adjusted["department"].astype(str).str.strip().str.casefold()
    bed_type = adjusted["bed_type"].astype(str).str.strip().str.casefold()
    status = adjusted["status"].astype(str).str.strip().str.casefold()
    icu = department.eq("icu") | bed_type.eq("icu")
    if delta < 0:
        available_icu = adjusted.index[
            icu & status.isin({"available", "vacant", "open", "ready", "clean", "empty"})
        ].tolist()
        if len(available_icu) < abs(delta):
            raise ValueError(
                f"Cannot remove {abs(delta)} ICU beds: only {len(available_icu)} "
                "available ICU beds exist in the planning inventory."
            )
        adjusted.loc[available_icu[:abs(delta)], "status"] = "Maintenance"
    elif delta > 0:
        template = adjusted.iloc[0].to_dict()
        added = []
        for index in range(delta):
            row = template.copy()
            row.update({
                "bed_id": f"SIM-ICU-{uuid4().hex[:8]}-{index + 1}",
                "department": "icu",
                "bed_type": "icu",
                "status": "Available",
                "isolation_capable": False,
                "available_for_emergency": False,
                "emergency_reserve": False,
                "scenario_only": True,
                "resource_source": "what_if_simulation",
            })
            added.append(row)
        adjusted = pd.concat([adjusted, pd.DataFrame(added)], ignore_index=True)
    return adjusted


def _recommendation(baseline: dict, scenario: dict) -> str:
    overflow = scenario["overflow_bed_days"]
    baseline_overflow = baseline["overflow_bed_days"]
    if overflow > baseline_overflow:
        return (
            f"The scenario adds {overflow - baseline_overflow} overflow bed-days versus "
            "the saved optimized baseline. Review the affected department/date rows and "
            "the staffing or bed-capacity constraints before operational use."
        )
    if scenario["unmet_isolation_bed_days"] > baseline["unmet_isolation_bed_days"]:
        return (
            "The scenario increases unmet isolation demand. Review isolation-capable "
            "inventory and the scenario assumptions."
        )
    if scenario["icu_overflow_bed_days"] > baseline["icu_overflow_bed_days"]:
        return (
            "The scenario increases ICU overflow. Review the ICU bed and staffing "
            "constraints before operational use."
        )
    return (
        "The solver reports no increase in the measured overflow or reserve violations "
        "for this scenario. This is a synthetic planning result, not a live operational order."
    )


def simulate_scenario(
    scenario_name: str,
    admissions_multiplier: float,
    icu_bed_delta: int,
    staff_availability_delta: float,
) -> dict[str, object]:
    from optimization import resource_optimization as planner

    manifest = read_manifest()
    if not manifest:
        raise FileNotFoundError("No completed portfolio run is available to simulate.")

    forecast_path = OUTPUT_DIR / "bed_availability_forecast.csv"
    inventory_path = planner.RESOURCE_SUPPORT / "bed_inventory_snapshot.csv"
    forecast = pd.read_csv(forecast_path)
    inputs = {"beds": pd.read_csv(inventory_path)}
    synthetic_surge_beds = planner.build_synthetic_surge_beds(forecast, inputs["beds"])
    if not synthetic_surge_beds.empty:
        inputs["beds"] = pd.concat(
            [inputs["beds"], synthetic_surge_beds],
            ignore_index=True,
        )

    if admissions_multiplier != 1.0:
        added_admissions = (
            (admissions_multiplier - 1.0)
            * pd.to_numeric(forecast["predicted_admissions"], errors="coerce").fillna(0)
            * pd.to_numeric(forecast["mean_predicted_los_days"], errors="coerce").fillna(0)
        )
        forecast["projected_occupied_beds"] = (
            pd.to_numeric(forecast["projected_occupied_beds"], errors="coerce")
            .fillna(0)
            + added_admissions
        ).clip(lower=0)
        forecast["projected_isolation_bed_demand"] = (
            pd.to_numeric(forecast["projected_isolation_bed_demand"], errors="coerce")
            .fillna(0)
            * admissions_multiplier
        ).clip(lower=0)

    inputs["beds"] = _apply_icu_capacity_delta(inputs["beds"], icu_bed_delta)
    coverage_path = OUTPUT_DIR / "staff_coverage_kpis.csv"
    if not coverage_path.is_file():
        raise FileNotFoundError(f"Required solver input is missing: {coverage_path}")
    staff_coverage = pd.read_csv(coverage_path)
    if staff_availability_delta != 0:
        assigned = pd.to_numeric(staff_coverage["assigned_staff"], errors="coerce").fillna(0)
        staff_coverage["assigned_staff"] = (assigned * (1 + staff_availability_delta)).apply(
            lambda value: max(0, int(value))
        )
        staff_coverage["unfilled_staff"] = (
            pd.to_numeric(staff_coverage["required_staff"], errors="coerce").fillna(0)
            - staff_coverage["assigned_staff"]
        ).clip(lower=0)

    allocation, scenario_kpis = planner.optimize_bed_allocations(
        forecast,
        inputs,
        staff_coverage,
    )
    emergency = planner.optimize_emergency_capacity(
        forecast,
        inputs,
        staff_coverage,
        allocation,
    )
    baseline_kpis = pd.read_csv(OUTPUT_DIR / "bed_allocation_kpis.csv")
    baseline_row = baseline_kpis[baseline_kpis["scenario"].eq("optimized")]
    if baseline_row.empty:
        raise ValueError("The saved optimized baseline KPI row is missing.")
    optimized_rows = scenario_kpis[scenario_kpis["scenario"].eq("optimized")]
    if optimized_rows.empty:
        raise ValueError("The scenario solver did not produce optimized KPI results.")
    scenario_row = optimized_rows.iloc[0].to_dict()
    baseline_row = baseline_row.iloc[0].to_dict()
    baseline_emergency = pd.read_csv(OUTPUT_DIR / "emergency_capacity_optimization.csv")

    def kpis(row: dict) -> dict[str, float | int]:
        return {
            "forecast_bed_days": int(round(float(row["forecast_bed_days"]))),
            "allocated_bed_days": int(round(float(row["allocated_bed_days"]))),
            "overflow_bed_days": int(row["overflow_bed_days"]),
            "overflow_due_to_staffing": int(row["overflow_bed_days_due_to_staffing"]),
            "overflow_due_to_bed_constraints": int(row["overflow_bed_days_due_to_bed_constraints"]),
            "unmet_isolation_bed_days": int(row["unmet_isolation_bed_days"]),
            "staffing_constraint_violations": int(row["staffing_constraint_violations"]),
            "physical_capacity_violations": int(row["physical_capacity_violations"]),
            "emergency_icu_reserve_violations": int(row["emergency_icu_reserve_violations"]),
        }

    baseline_metrics = kpis(baseline_row)
    scenario_metrics = kpis(scenario_row)
    scenario_metrics["icu_overflow_bed_days"] = int(
        pd.to_numeric(emergency["icu_overflow"], errors="coerce").fillna(0).sum()
    )
    baseline_metrics["icu_overflow_bed_days"] = int(
        pd.to_numeric(baseline_emergency["icu_overflow"], errors="coerce").fillna(0).sum()
    )
    baseline_overview = get_overview()
    scenario_metrics.update({
        # Counts are presented as whole numbers (half up); the solver
        # works on the unrounded projections.
        "projected_admissions": int(
            float(baseline_overview["predicted_admissions"]) * admissions_multiplier + 0.5
        ),
        # Hospital-wide: sum departments per day, then take the peak day.
        "peak_projected_occupied_beds": int(
            float(forecast.groupby("date")["projected_occupied_beds"].sum().max()) + 0.5
        ),
        "icu_bed_delta": icu_bed_delta,
        "staff_availability_delta": staff_availability_delta,
        "unfilled_staff_positions": int(
            pd.to_numeric(staff_coverage["unfilled_staff"], errors="coerce").fillna(0).sum()
        ),
    })

    simulation_id = str(uuid4())
    timestamp = datetime.now(timezone.utc)
    assumptions = {
        "admissions_multiplier": admissions_multiplier,
        "icu_bed_delta": icu_bed_delta,
        "staff_availability_delta": staff_availability_delta,
    }
    recommendation = _recommendation(baseline_metrics, scenario_metrics)
    with SessionLocal() as session:
        session.add(SimulationRecord(
            id=simulation_id,
            scenario=scenario_name,
            assumptions=assumptions,
            baseline_metrics=baseline_metrics,
            simulated_metrics=scenario_metrics,
            created_at=timestamp.replace(tzinfo=None),
        ))
        for metric, baseline_value in baseline_metrics.items():
            scenario_value = scenario_metrics.get(metric)
            session.add(SimulationResult(
                simulation_id=simulation_id,
                metric_name=metric,
                baseline_value=float(baseline_value),
                simulated_value=(
                    float(scenario_value) if isinstance(scenario_value, (float, int)) else None
                ),
            ))
        session.commit()

    return {
        "simulation_id": simulation_id,
        "scenario_name": scenario_name,
        "assumptions": assumptions,
        "baseline_metrics": baseline_metrics,
        "simulated_metrics": scenario_metrics,
        "operational_recommendation": recommendation,
        "timestamp": timestamp.isoformat(),
        "source_run_id": str(manifest.get("scenario_as_of_date", "unknown")),
        "solver": "OR-Tools CP-SAT",
        "scope_note": (
            "The optimizer is rerun on generated forecast and inventory inputs. "
            "Additional admissions are estimated as extra arrivals multiplied by "
            "the existing department mean predicted length of stay. Simulation results "
            "remain synthetic scenarios, not live operational orders."
        ),
    }


def get_simulation(simulation_id: str) -> dict[str, object] | None:
    with SessionLocal() as session:
        record = session.get(SimulationRecord, simulation_id)
        if record is None:
            return None
        return {
            "simulation_id": record.id,
            "scenario_name": record.scenario,
            "assumptions": record.assumptions,
            "baseline_metrics": record.baseline_metrics,
            "simulated_metrics": record.simulated_metrics,
            "timestamp": record.created_at.isoformat() if record.created_at else None,
        }


def get_simulation_runs(limit: int = 100) -> list[dict[str, object]]:
    with SessionLocal() as session:
        records = (
            session.query(SimulationRecord)
            .order_by(SimulationRecord.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "simulation_id": record.id,
                "scenario_name": record.scenario,
                "assumptions": record.assumptions,
                "baseline_metrics": record.baseline_metrics,
                "simulated_metrics": record.simulated_metrics,
                "timestamp": record.created_at.isoformat() if record.created_at else None,
            }
            for record in records
        ]
