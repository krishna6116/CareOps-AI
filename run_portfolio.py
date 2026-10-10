"""Build and validate a complete, isolated synthetic CareOps AI scenario."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SYNTHETIC_DATA = ROOT / "data" / "synthetic"
OUTPUT = ROOT / "optimization" / "resource_planning" / "outputs"
SOURCE_FILES = {
    "patients.csv": {"patient_id", "patient_name", "age", "gender"},
    "admissions.csv": {
        "admission_id", "patient_id", "patient_name", "department", "admission_date",
        "discharge_date", "LOS", "ICU_required", "assigned_bed_id", "stay_status",
    },
    "beds.csv": {
        "bed_id", "department", "bed_type", "status", "isolation_capable",
        "emergency_reserve", "available_for_emergency",
    },
    "staff.csv": {"staff_id", "department", "role", "availability", "max_hours"},
    "appointments.csv": {
        "appointment_id", "doctor_id", "department", "date", "time", "duration",
        "no_show",
    },
    "daily_operations.csv": {
        "date", "department", "admissions", "discharges", "ICU_admissions",
        "occupancy", "staff_demand",
    },
    "shifts.csv": {"shift_id", "date", "staff_id", "shift", "hours", "assigned"},
    "hospital_snapshot.csv": {"snapshot_date", "total_beds", "occupied_beds"},
}
FORECASTS = {
    "admission_forecasting": ROOT / "ml" / "admission_forecasting" / "admission_30_day_forecast.csv",
    "occupancy_forecasting": ROOT / "ml" / "occupancy_forecasting" / "occupancy_future_forecast.csv",
    "icu_forecasting": ROOT / "ml" / "icu_demand_forecasting" / "icu_future_forecast.csv",
}
PIPELINE = [
    ROOT / "data" / "synthetic" / "generate_data.py",
    ROOT / "ml" / "admission_forecasting.py",
    ROOT / "ml" / "los_prediction" / "los_prediction.py",
    ROOT / "ml" / "occupancy_forecasting.py",
    ROOT / "ml" / "icu_demand_forecasting" / "icu_demand_forecasting.py",
    ROOT / "ml" / "no_show_prediction" / "no_show_prediction.py",
    ROOT / "optimization" / "resource_optimization.py",
    ROOT / "reliability_scorecard.py",
]
# Exploratory stages: cleaned copies, EDA reports, and feature tables built
# from the same synthetic source. The forecasting models build their own
# leakage-safe features and do not read these outputs.
EXPLORATORY = [
    ROOT / "data" / "data_cleaning.py",
    ROOT / "data" / "eda_complete.py",
    ROOT / "data" / "feature_engineering.py",
]


def build_pipeline_readiness(
    scorecard: dict,
    manifest: dict,
    no_show_metadata: dict,
) -> dict:
    """Return an explicit handoff status without promoting synthetic results."""
    minimum_target = float(scorecard["minimum_target"])
    sections = scorecard["sections"]
    below_target = [
        item["section"] for item in sections
        if float(item["score"]) < minimum_target
    ]
    no_show_section = next(
        (
            item for item in sections
            if item["section"] == "No-show prediction"
        ),
        None,
    )
    if no_show_section is None:
        raise RuntimeError("Reliability scorecard is missing No-show prediction.")

    no_show_metadata_ok = (
        bool(no_show_metadata.get("historical_features_are_leakage_safe"))
        and no_show_metadata.get("source_data_modified") is False
        and float(no_show_section["score"]) >= 4.5
    )
    capacity_gates = manifest["optimized_capacity_gates"]
    failed_capacity_gates = [
        name for name, value in capacity_gates.items() if int(value) != 0
    ]
    structural_ok = (
        bool(manifest["source_integrity_checks_passed"])
        and int(manifest["audit_failures"]) == 0
    )
    blockers = [
        f"Reliability section below {minimum_target}/5: {name}"
        for name in below_target
    ]
    if not no_show_metadata_ok:
        blockers.append(
            "No-show model must retain leakage-safe features, unchanged source "
            "data, and a reliability score of at least 4.5/5."
        )
    if failed_capacity_gates:
        blockers.append(
            "Capacity gates failed: " + ", ".join(failed_capacity_gates)
        )
    if not structural_ok:
        blockers.append("Source integrity or resource audit checks failed.")

    return {
        "pipeline_status": (
            "ready_for_next_process" if not blockers
            else "completed_review_required"
        ),
        "next_process_ready": not blockers,
        "next_process_blockers": blockers,
        "review_required_sections": below_target,
        "no_show_model_integration": {
            "model_version": no_show_metadata.get("model_version"),
            "historical_features_are_leakage_safe": bool(
                no_show_metadata.get("historical_features_are_leakage_safe")
            ),
            "source_data_modified": no_show_metadata.get("source_data_modified"),
            "score": float(no_show_section["score"]),
            "required_score": 4.5,
        },
        "ready_for_live_operational_use": False,
    }


def validate_source_data(
    as_of: date,
    source_dir: Path | None = None,
) -> dict[str, int]:
    source_dir = source_dir or SYNTHETIC_DATA
    row_counts: dict[str, int] = {}
    frames: dict[str, pd.DataFrame] = {}
    for filename, required_columns in SOURCE_FILES.items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Synthetic source dataset was not generated: {path}")
        frame = pd.read_csv(path)
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"{filename} contains no synthetic records.")
        row_counts[filename] = len(frame)
        frames[filename.removesuffix(".csv")] = frame

    patients = frames["patients"]
    admissions = frames["admissions"]
    beds = frames["beds"]
    staff = frames["staff"]
    appointments = frames["appointments"]
    shifts = frames["shifts"]
    operations = frames["daily_operations"]

    for name, frame, key in (
        ("patients", patients, "patient_id"),
        ("admissions", admissions, "admission_id"),
        ("beds", beds, "bed_id"),
        ("staff", staff, "staff_id"),
        ("appointments", appointments, "appointment_id"),
        ("shifts", shifts, "shift_id"),
    ):
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Synthetic {name} identifiers must be present and unique.")

    operations["date"] = pd.to_datetime(operations["date"], errors="coerce")
    if operations["date"].isna().any():
        raise ValueError("Synthetic operations include invalid dates.")
    latest_operation = operations["date"].max().date()
    if latest_operation != as_of:
        raise ValueError(
            f"Synthetic operations end on {latest_operation}, not requested scenario date {as_of}."
        )
    if operations.duplicated(["date", "department"]).any():
        raise ValueError("Synthetic operations contain duplicate date/department rows.")
    department_counts = operations.groupby("date")["department"].nunique()
    if department_counts.nunique() != 1:
        raise ValueError("Synthetic operations have inconsistent department coverage by day.")

    operation_dates = pd.DatetimeIndex(operations["date"].dt.normalize().unique()).sort_values()
    expected_operations_dates = pd.date_range(
        pd.Timestamp(operations["date"].min()).normalize(),
        pd.Timestamp(as_of),
        freq="D",
    )
    if not operation_dates.equals(expected_operations_dates):
        raise ValueError("Synthetic operations do not cover every date through the scenario cutoff.")

    patients_by_id = patients.set_index("patient_id")
    if not admissions["patient_id"].isin(patients_by_id.index).all():
        raise ValueError("Synthetic admissions reference unknown patient IDs.")
    admission_patient_rows = patients_by_id.loc[admissions["patient_id"]]
    if not admissions["patient_name"].reset_index(drop=True).equals(
        admission_patient_rows["patient_name"].reset_index(drop=True)
    ):
        raise ValueError("Admission patient names do not match their patient records.")

    admission_dates = pd.to_datetime(admissions["admission_date"], errors="coerce")
    discharge_dates = pd.to_datetime(admissions["discharge_date"], errors="coerce")
    lengths_of_stay = pd.to_numeric(admissions["LOS"], errors="coerce")
    if (
        admission_dates.isna().any()
        or discharge_dates.isna().any()
        or lengths_of_stay.isna().any()
        or (lengths_of_stay <= 0).any()
        or (discharge_dates <= admission_dates).any()
        or ((discharge_dates - admission_dates).dt.days != lengths_of_stay).any()
        or (admission_dates.dt.date > as_of).any()
    ):
        raise ValueError("Synthetic admission dates and LOS must form valid positive stays.")
    if "stay_status" not in admissions.columns:
        raise ValueError("Synthetic admissions must label completed and in-progress stays.")
    expected_stay_status = pd.Series(
        ["completed" if value.date() <= as_of else "in_progress" for value in discharge_dates],
        index=admissions.index,
    )
    if not admissions["stay_status"].astype(str).str.casefold().equals(expected_stay_status):
        raise ValueError("Synthetic stay_status values disagree with the scenario cutoff.")

    icu_flags = pd.to_numeric(admissions["ICU_required"], errors="coerce")
    if icu_flags.isna().any() or not icu_flags.isin([0, 1]).all():
        raise ValueError("Synthetic ICU requirements must be binary values.")
    if admissions["assigned_bed_id"].isna().any():
        raise ValueError("Every capacity-feasible synthetic admission must have an assigned bed.")
    bed_lookup = beds.set_index("bed_id")
    emergency_eligible = beds["available_for_emergency"].astype(str).str.casefold().isin(
        {"1", "true", "yes", "y"}
    )
    if (
        (emergency_eligible & beds["department"].ne("General Medicine")).any()
        or (
            emergency_eligible
            & beds["status"].astype(str).str.casefold().ne("available")
        ).any()
    ):
        raise ValueError(
            "Only available General Medicine beds may be marked emergency-flexible."
        )
    if not admissions["assigned_bed_id"].isin(bed_lookup.index).all():
        raise ValueError("Synthetic admissions reference unknown beds.")
    assigned_beds = bed_lookup.loc[admissions["assigned_bed_id"]]
    expected_bed_department = admissions["department"].where(
        icu_flags.eq(0), "ICU"
    ).reset_index(drop=True)
    actual_bed_department = assigned_beds["department"].reset_index(drop=True)
    if not actual_bed_department.equals(expected_bed_department):
        raise ValueError("Admission bed assignments do not match department/ICU requirements.")
    expected_bed_type = admissions["department"].map({
        "Emergency": "Emergency",
        "Oncology": "Isolation",
    }).fillna("General")
    expected_bed_type = expected_bed_type.where(icu_flags.eq(0), "ICU")
    if not assigned_beds["bed_type"].reset_index(drop=True).equals(
        expected_bed_type.reset_index(drop=True)
    ):
        raise ValueError("Synthetic admissions reference an incompatible bed type.")

    bed_intervals = admissions.assign(
        _admission_date=admission_dates,
        _discharge_date=discharge_dates,
    ).sort_values(["assigned_bed_id", "_admission_date", "_discharge_date"])
    prior_discharge = bed_intervals.groupby("assigned_bed_id")["_discharge_date"].shift()
    if (prior_discharge > bed_intervals["_admission_date"]).any():
        raise ValueError("Synthetic bed assignments contain overlapping patient stays.")

    staff_by_id = staff.set_index("staff_id")
    if not appointments["patient_id"].isin(patients_by_id.index).all():
        raise ValueError("Synthetic appointments reference unknown patient IDs.")
    if not appointments["doctor_id"].isin(staff_by_id.index).all():
        raise ValueError("Synthetic appointments reference unknown staff IDs.")
    appointment_staff = staff_by_id.loc[appointments["doctor_id"]]
    if not appointment_staff["role"].isin(["Doctor", "Surgeon"]).all():
        raise ValueError("Synthetic appointments must be assigned to doctors or surgeons.")
    if not appointment_staff["department"].reset_index(drop=True).equals(
        appointments["department"].reset_index(drop=True)
    ):
        raise ValueError("Synthetic appointments are not department-matched to their providers.")
    appointment_dates = pd.to_datetime(
        appointments["date"].astype(str) + " " + appointments["time"].astype(str),
        errors="coerce",
    )
    appointment_durations = pd.to_numeric(appointments["duration"], errors="coerce")
    if (
        appointment_dates.isna().any()
        or appointment_durations.isna().any()
        or (appointment_durations <= 0).any()
        or not pd.to_numeric(appointments["no_show"], errors="coerce").isin([0, 1]).all()
    ):
        raise ValueError("Synthetic appointment dates, durations, and outcomes are invalid.")

    if not shifts["staff_id"].isin(staff_by_id.index).all():
        raise ValueError("Synthetic shifts reference unknown staff IDs.")
    shift_dates = pd.to_datetime(shifts["date"], errors="coerce")
    shift_hours = pd.to_numeric(shifts["hours"], errors="coerce")
    if shift_dates.isna().any() or shift_hours.isna().any() or (shift_hours <= 0).any():
        raise ValueError("Synthetic shifts must have valid dates and positive hours.")
    if shifts.duplicated(["staff_id", "date"]).any():
        raise ValueError("Synthetic staff cannot have more than one shift per day.")
    shift_staff = staff_by_id.loc[shifts["staff_id"]]
    max_hours = pd.to_numeric(shift_staff["max_hours"], errors="coerce").to_numpy()
    shift_frame = shifts.assign(
        _date=shift_dates,
        _hours=shift_hours,
        _max_hours=max_hours,
    )
    for _, staff_shifts in shift_frame.groupby("staff_id"):
        rolling_hours = (
            staff_shifts.sort_values("_date")
            .set_index("_date")["_hours"]
            .rolling("7D", closed="right")
            .sum()
        )
        if (rolling_hours.to_numpy() > staff_shifts["_max_hours"].iloc[0]).any():
            raise ValueError("Synthetic staff shifts exceed the rolling weekly hour limits.")
    ordered_shifts = shift_frame.sort_values(["staff_id", "_date"])
    prior_shift = ordered_shifts.groupby("staff_id")["shift"].shift()
    prior_shift_date = ordered_shifts.groupby("staff_id")["_date"].shift()
    rest_violation = (
        prior_shift.eq("Night")
        & ordered_shifts["shift"].eq("Morning")
        & (ordered_shifts["_date"] - prior_shift_date).dt.days.eq(1)
    )
    if rest_violation.any():
        raise ValueError("Synthetic schedules contain a night-to-morning rest violation.")

    # Verify that daily source counts and bed census are derived from the same encounters.
    operation_keys = ["date", "department"]
    admission_counts = admissions.assign(
        date=admission_dates.dt.normalize()
    ).groupby(operation_keys).size().rename("_admissions")
    discharge_counts = admissions.assign(
        date=discharge_dates.dt.normalize()
    ).groupby(operation_keys).size().rename("_discharges")
    icu_counts = admissions.loc[icu_flags.eq(1)].assign(
        date=admission_dates.loc[icu_flags.eq(1)].dt.normalize()
    ).groupby(operation_keys).size().rename("_icu_admissions")
    operation_counts = operations.set_index(operation_keys)
    for expected, column in (
        (admission_counts, "admissions"),
        (discharge_counts, "discharges"),
        (icu_counts, "ICU_admissions"),
    ):
        aligned = expected.reindex(operation_counts.index, fill_value=0)
        actual = pd.to_numeric(operation_counts[column], errors="coerce")
        if actual.isna().any() or not actual.reset_index(drop=True).equals(
            aligned.reset_index(drop=True).astype(actual.dtype)
        ):
            raise ValueError(f"Synthetic operations {column} do not match admission events.")

    bed_events = admissions.assign(
        _bed_department=assigned_beds["department"].to_numpy(),
        _admission_day=admission_dates.dt.normalize(),
        _discharge_day=discharge_dates.dt.normalize(),
    )
    capacities = beds.groupby("department").size().to_dict()
    census_records = []
    for department in beds["department"].unique():
        department_events = bed_events[bed_events["_bed_department"] == department]
        arrived = department_events.groupby("_admission_day").size()
        discharged = department_events.groupby("_discharge_day").size()
        census = (
            arrived.reindex(operation_dates, fill_value=0)
            - discharged.reindex(operation_dates, fill_value=0)
        ).cumsum()
        census_records.extend(
            (day, department, int(value)) for day, value in census.items()
        )
    census_frame = pd.DataFrame(
        census_records, columns=["date", "department", "_census"]
    )
    operation_census = operations.merge(
        census_frame, on=["date", "department"], how="left", validate="one_to_one"
    )
    if operation_census["_census"].isna().any() or not np.array_equal(
        operation_census["bed_demand"].to_numpy(dtype=int),
        operation_census["_census"].to_numpy(dtype=int),
    ):
        raise ValueError("Synthetic operations census does not match active bed assignments.")
    expected_occupancy = (
        operation_census["_census"]
        / operation_census["department"].map(capacities)
        * 100
    ).round(2)
    if not np.allclose(
        pd.to_numeric(operation_census["occupancy"], errors="coerce"),
        expected_occupancy,
        atol=0.01,
    ):
        raise ValueError("Synthetic occupancy rates do not match bed census and capacity.")

    snapshot = frames["hospital_snapshot"]
    if len(snapshot) != 1 or pd.to_datetime(
        snapshot.loc[0, "snapshot_date"], errors="coerce"
    ).date() != as_of:
        raise ValueError("Synthetic hospital snapshot must be a single row for the cutoff date.")
    status_counts = beds["status"].value_counts()
    if (
        int(snapshot.loc[0, "total_beds"]) != len(beds)
        or int(snapshot.loc[0, "occupied_beds"]) != int(status_counts.get("Occupied", 0))
        or int(snapshot.loc[0, "available_beds"]) != int(status_counts.get("Available", 0))
        or int(snapshot.loc[0, "maintenance_beds"]) != int(status_counts.get("Maintenance", 0))
        or int(snapshot.loc[0, "total_icu_beds"]) != int(beds["bed_type"].eq("ICU").sum())
        or int(snapshot.loc[0, "occupied_icu_beds"]) != int(
            ((beds["bed_type"] == "ICU") & (beds["status"] == "Occupied")).sum()
        )
    ):
        raise ValueError("Synthetic hospital snapshot does not reconcile to bed inventory.")

    return row_counts


def validate_forecast_artifacts(as_of: date) -> dict[str, list[str]]:
    expected = pd.date_range(pd.Timestamp(as_of) + pd.Timedelta(days=1), periods=30, freq="D")
    date_summary: dict[str, list[str]] = {}
    for name, path in FORECASTS.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} did not produce its expected forecast: {path}")
        forecast = pd.read_csv(path)
        if "date" not in forecast.columns:
            raise ValueError(f"{name} forecast is missing its date column.")
        actual = pd.DatetimeIndex(
            pd.to_datetime(forecast["date"], errors="coerce").dropna().dt.normalize().unique()
        ).sort_values()
        if not actual.equals(expected):
            raise ValueError(
                f"{name} forecast dates do not exactly cover the 30-day window "
                f"{expected[0].date()} through {expected[-1].date()}."
            )
        numeric_columns = {
            "admission_forecasting": ["predicted_admissions"],
            "occupancy_forecasting": ["forecast_occupied_beds"],
            "icu_forecasting": ["predicted_icu_admissions"],
        }[name]
        missing_values = set(numeric_columns) - set(forecast.columns)
        if missing_values:
            raise ValueError(
                f"{name} forecast is missing required measures: "
                f"{sorted(missing_values)}."
            )
        values = forecast[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or (values < 0).any().any():
            raise ValueError(f"{name} forecast contains invalid or negative values.")
        if forecast["date"].isna().any():
            raise ValueError(f"{name} forecast contains invalid dates.")
        if name == "admission_forecasting":
            if "department" not in forecast.columns:
                raise ValueError("Admission forecast is missing department grain.")
            if forecast.duplicated(["date", "department"]).any():
                raise ValueError("Admission forecast contains duplicate date/department rows.")
        elif len(actual) != len(forecast) or forecast["date"].duplicated().any():
            raise ValueError(f"{name} forecast contains duplicate dates.")
        if name == "icu_forecasting":
            if set(forecast.get("forecast_target_unit", pd.Series(dtype=str)).astype(str)) != {
                "admissions_per_day"
            }:
                raise ValueError("ICU forecast target must be daily admissions, not census.")
        date_summary[name] = [actual.min().date().isoformat(), actual.max().date().isoformat()]
    return date_summary


def validate_appointment_schedule(schedule: pd.DataFrame) -> int:
    if schedule.empty or "status" not in schedule.columns:
        raise ValueError("Optimized appointment schedule is empty or missing status.")
    unscheduled = schedule[schedule["status"] != "SCHEDULED"]
    if not unscheduled.empty:
        raise ValueError(
            f"Optimized appointment schedule has {len(unscheduled)} unassigned appointments."
        )
    return len(schedule)


def run_pipeline(as_of: date, postgres: bool = False, exploratory: bool = False) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT / "portfolio_run_manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    environment = os.environ.copy()
    environment["CAREOPS_AS_OF_DATE"] = as_of.isoformat()
    environment["CAREOPS_DATA_DIR"] = str(SYNTHETIC_DATA)

    steps = list(PIPELINE)
    if exploratory:
        steps[1:1] = EXPLORATORY
    for index, script in enumerate(steps, start=1):
        print(f"\n[{index}/{len(steps)}] Running {script.relative_to(ROOT)}", flush=True)
        subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT,
            env=environment,
            check=True,
        )
        if script.name == "generate_data.py":
            row_counts = validate_source_data(as_of)
            print(f"Validated generated synthetic source rows: {row_counts}", flush=True)
        if script.name == "resource_optimization.py":
            forecast_dates = validate_forecast_artifacts(as_of)
            appointment_schedule = pd.read_csv(
                OUTPUT / "optimized_appointment_schedule.csv"
            )
            scheduled_appointments = validate_appointment_schedule(
                appointment_schedule
            )
            audit_path = OUTPUT / "data_quality_audit.csv"
            audit = pd.read_csv(audit_path)
            failures = audit[audit["status"] == "FAIL"]
            if not failures.empty:
                names = ", ".join(failures["check"].astype(str))
                raise RuntimeError(f"Resource input audit failed checks: {names}")
            allocation = pd.read_csv(OUTPUT / "bed_allocation_kpis.csv")
            optimized_allocation = allocation[allocation["scenario"] == "optimized"]
            coverage = pd.read_csv(OUTPUT / "staff_coverage_kpis.csv")
            emergency = pd.read_csv(OUTPUT / "emergency_capacity_optimization.csv")
            if int(optimized_allocation["overflow_bed_days"].sum()) != 0:
                raise RuntimeError("Optimized portfolio scenario has remaining bed overflow.")
            if int(coverage["unfilled_staff"].sum()) != 0:
                raise RuntimeError("Portfolio scenario has unfilled staff positions.")
            if int(emergency["icu_overflow"].sum()) != 0:
                raise RuntimeError("Portfolio scenario has remaining ICU overflow.")
            if int(emergency["emergency_overflow"].sum()) != 0:
                raise RuntimeError("Portfolio scenario has remaining emergency overflow.")
            capacity_plan = pd.read_csv(OUTPUT / "capacity_expansion_plan.csv")
            bed_plan = capacity_plan[
                capacity_plan["resource_type"] == "temporary_surge_beds"
            ]
            staff_plan = capacity_plan[
                capacity_plan["resource_type"] == "synthetic_surge_staff"
            ]
            (OUTPUT / "portfolio_run_manifest.json").write_text(
                json.dumps({
                    "project": "CareOps-AI",
                    "data_scope": "fully_synthetic_portfolio_scenario",
                    "scenario_as_of_date": as_of.isoformat(),
                    "forecast_horizon_days": 30,
                    "forecast_windows": forecast_dates,
                    "synthetic_source_row_counts": row_counts,
                    "source_integrity_checks_passed": True,
                    "audit_failures": 0,
                    "audit_warnings": audit.loc[
                        audit["status"] == "WARN", "check"
                    ].astype(str).tolist(),
                    "optimized_capacity_gates": {
                        "overflow_bed_days": 0,
                        "unfilled_staff_positions": 0,
                        "icu_overflow_bed_days": 0,
                        "emergency_overflow_bed_days": 0,
                        "unassigned_appointments": 0,
                    },
                    "optimized_appointment_count": scheduled_appointments,
                    "synthetic_capacity_expansion": {
                        "scenario_only": True,
                        "temporary_surge_beds": int(bed_plan["added_quantity"].sum()),
                        "surge_staff_pool_headcount": int(staff_plan["added_quantity"].sum()),
                        "scheduled_surge_hours": float(staff_plan["scheduled_hours"].sum()),
                        "illustrative_expansion_cost_usd": float(
                            capacity_plan["planned_cost_usd"].sum()
                        ),
                        "costs_are_scenario_assumptions_not_market_quotes": True,
                    },
                    "not_for_clinical_or_live_operational_use": True,
                }, indent=2) + "\n",
                encoding="utf-8",
            )
        if script.name == "reliability_scorecard.py":
            scorecard_path = OUTPUT / "reliability_scorecard.json"
            if not scorecard_path.is_file():
                raise RuntimeError("Reliability scorecard was not generated.")

    manifest_path = OUTPUT / "portfolio_run_manifest.json"
    scorecard_path = OUTPUT / "reliability_scorecard.json"
    no_show_metadata_path = (
        ROOT / "ml" / "no_show_prediction" / "no_show_model_metadata.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    no_show_metadata = json.loads(
        no_show_metadata_path.read_text(encoding="utf-8")
    )
    readiness = build_pipeline_readiness(
        scorecard, manifest, no_show_metadata
    )
    manifest.update(readiness)
    if postgres:
        if readiness["next_process_ready"]:
            from optimization.postgresql_integration import (
                load_pipeline_to_postgres,
            )

            try:
                manifest["postgresql_integration"] = load_pipeline_to_postgres(
                    ROOT,
                    manifest,
                    os.environ.get("CAREOPS_DATABASE_URL"),
                )
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                manifest["postgresql_integration"] = {
                    "status": "failed",
                    "schema": "careops",
                    "error": str(exc),
                }
                manifest["next_process_ready"] = False
                manifest["pipeline_status"] = "completed_review_required"
                manifest["next_process_blockers"].append(
                    f"PostgreSQL handoff failed: {exc}"
                )
        else:
            manifest["postgresql_integration"] = {
                "status": "blocked",
                "schema": "careops",
                "error": "Synthetic pipeline readiness gates did not pass.",
            }
            manifest["next_process_ready"] = False
            manifest["pipeline_status"] = "completed_review_required"
            manifest["next_process_blockers"].append(
                "PostgreSQL handoff blocked until pipeline readiness gates pass."
            )
    manifest["reliability_scorecard"] = {
        **manifest.get("reliability_scorecard", {}),
        "path": str(scorecard_path),
        "minimum_target": scorecard["minimum_target"],
        "overall_score": scorecard["overall_score"],
        "all_sections_at_or_above_target": scorecard[
            "all_sections_at_or_above_target"
        ],
        "data_asset_count": len(scorecard["data_assets"]),
        "forecast_score_count": len(scorecard["forecasts"]),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    subprocess.run(
        [sys.executable, str(ROOT / "ml" / "model_metrics_report.py")],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate the synthetic dataset, train all forecasts, and optimize resources."
    )
    parser.add_argument(
        "--as-of-date",
        type=date.fromisoformat,
        default=date.today(),
        help="Scenario cutoff in YYYY-MM-DD format (default: today).",
    )
    parser.add_argument(
        "--postgres",
        action="store_true",
        help=(
            "Load validated source, forecast, and optimization artifacts into "
            "the careops PostgreSQL schema. Requires CAREOPS_DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help=(
            "Also rebuild data/processed, the EDA reports, and the exploratory "
            "feature tables from the generated synthetic data."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.as_of_date, postgres=args.postgres, exploratory=args.exploratory)
    manifest = json.loads(
        (OUTPUT / "portfolio_run_manifest.json").read_text(encoding="utf-8")
    )
    print(f"\nPortfolio scenario completed for {args.as_of_date}.", flush=True)
    print(f"Validated outputs: {OUTPUT}", flush=True)
    print(f"Pipeline status: {manifest['pipeline_status']}", flush=True)
    if not manifest["next_process_ready"]:
        print("Next process is blocked:", flush=True)
        for blocker in manifest["next_process_blockers"]:
            print(f" - {blocker}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
