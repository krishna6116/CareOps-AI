"""Hospital resource planning for bed, staff, appointment, and emergency capacity.

This is a synthetic-data planning aid, not an autonomous clinical decision system.
It reads the existing processed/forecast CSVs and writes scenario outputs beneath
``optimization/resource_planning/outputs``.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from ortools.sat.python import cp_model

if __package__:
    from .resource_validation import audit_resource_inputs, build_forecast_backtest
else:
    from resource_validation import audit_resource_inputs, build_forecast_backtest


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = Path(os.environ.get("CAREOPS_DATA_DIR", ROOT / "data" / "processed"))
ADMISSION_FORECAST = ROOT / "ml" / "admission_forecasting" / "admission_30_day_forecast.csv"
LOS_PREDICTIONS = ROOT / "ml" / "los_prediction" / "los_predictions.csv"
ICU_FORECAST = ROOT / "ml" / "icu_demand_forecasting" / "icu_future_forecast.csv"
OCCUPANCY_FORECAST = ROOT / "ml" / "occupancy_forecasting" / "occupancy_future_forecast.csv"
NO_SHOW_PREDICTIONS = ROOT / "ml" / "no_show_prediction" / "no_show_predictions.csv"
OUTPUT = ROOT / "optimization" / "resource_planning" / "outputs"
RESOURCE_SUPPORT = ROOT / "data" / "synthetic" / "resource_support"
RESOURCE_SUPPORT_GENERATOR = RESOURCE_SUPPORT / "generate_resource_support.py"

SHIFT_HOURS = 8
NURSE_TO_BED_CAPACITY = 4
EMERGENCY_RESERVE = 2
ICU_RESERVE = 2
ISOLATION_RESERVE = 1
SOLVER_SECONDS = 20
SOLVER_WORKERS = 1
SOLVER_RANDOM_SEED = 42
SHIFTS = {
    "Morning": (7 * 60, 15 * 60),
    "Evening": (15 * 60, 23 * 60),
    "Night": (23 * 60, 7 * 60),
}
CLINICAL_ROLE_TOKENS = (
    "nurse", "doctor", "physician", "surgeon", "clinician",
    "therapist", "paramedic", "technician", "practitioner",
)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required planning input not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def normalize(value: Any) -> str:
    return " ".join(str(value).strip().casefold().replace("_", " ").split())


def is_true(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return normalize(value) in {"1", "true", "yes", "y"}


def as_date(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def department_bed_type(department: str) -> str:
    key = normalize(department)
    return {
        "general medicine": "general",
        "icu": "icu",
        "emergency": "emergency",
    }.get(key, key)


def staffed_bed_capacity(
    coverage: pd.DataFrame,
    department: str,
    day: str,
) -> int:
    """Estimate round-the-clock bed capacity from the least-staffed nurse shift."""
    rows = coverage
    if "date" in rows.columns:
        rows = rows[rows["date"].astype(str) == str(day)]
    rows = rows[rows["department"].map(normalize) == normalize(department)]
    if rows.empty:
        return 0
    if "role" not in rows.columns or "shift" not in rows.columns:
        return int(pd.to_numeric(rows["assigned_staff"], errors="coerce").fillna(0).sum()) * NURSE_TO_BED_CAPACITY
    nurses = rows[rows["role"].map(normalize).str.contains("nurse", na=False)]
    if nurses.empty:
        return 0
    per_shift = nurses.groupby("shift")["assigned_staff"].sum()
    return min(int(per_shift.get(shift, 0)) for shift in SHIFTS) * NURSE_TO_BED_CAPACITY


def build_synthetic_surge_beds(
    forecast: pd.DataFrame,
    beds: pd.DataFrame,
) -> pd.DataFrame:
    """Size explicitly synthetic surge beds to the forecast peak and protected reserve."""
    operational_statuses = {
        "occupied", "in use", "available", "vacant", "open", "ready", "clean", "empty",
    }
    operational = beds[beds["status"].map(normalize).isin(operational_statuses)].copy()
    operational["_department"] = operational["department"].map(normalize)
    operational["_isolation"] = (
        operational["isolation_capable"].map(is_true)
        if "isolation_capable" in operational.columns
        else pd.Series(False, index=operational.index)
    )
    existing_capacity = operational.groupby("_department").size().to_dict()
    existing_isolation = (
        operational[operational["_isolation"]]
        .groupby("_department")
        .size()
        .to_dict()
    )
    rows: list[dict[str, Any]] = []
    for department, group in forecast.groupby("department"):
        key = normalize(department)
        peak_demand = int(math.ceil(
            pd.to_numeric(group["projected_occupied_beds"], errors="coerce")
            .fillna(0).max()
        ))
        reserve = (
            ICU_RESERVE if key == "icu"
            else EMERGENCY_RESERVE if key == "emergency"
            else 0
        )
        if "projected_isolation_bed_demand" in group:
            peak_isolation = int(math.ceil(
                pd.to_numeric(group["projected_isolation_bed_demand"], errors="coerce")
                .fillna(0).max()
            ))
        else:
            peak_isolation = 0
        capacity_gap = peak_demand + reserve - int(existing_capacity.get(key, 0))
        isolation_gap = peak_isolation + reserve - int(existing_isolation.get(key, 0))
        additions = max(0, capacity_gap, isolation_gap)
        for index in range(additions):
            bed_type = department_bed_type(key)
            rows.append({
                "bed_id": f"SURGE-{key.replace(' ', '-')}-{index + 1:03d}",
                "department": key,
                "bed_type": bed_type,
                "status": "Available",
                "isolation_capable": True,
                "available_for_emergency": key == "general medicine",
                "emergency_reserve": False,
                "as_of_date": str(pd.to_datetime(group["date"]).min().date()),
                "data_provenance": "synthetic_forecast_sized_surge_capacity",
                "scenario_only": True,
                "resource_source": "temporary_surge_bed_scenario",
            })
    return pd.DataFrame(rows)


def load_inputs() -> dict[str, pd.DataFrame]:
    inputs = {
        "beds": read_csv(RESOURCE_SUPPORT / "bed_inventory_snapshot.csv"),
        "staff": read_csv(RESOURCE_SUPPORT / "staff_availability_snapshot.csv"),
        "shifts": read_csv(PROCESSED / "shifts.csv"),
        "appointments": read_csv(PROCESSED / "appointments.csv"),
        "admissions": read_csv(PROCESSED / "admissions.csv"),
        "isolation_labels": read_csv(RESOURCE_SUPPORT / "admission_isolation_labels.csv"),
        "isolation_forecast": read_csv(RESOURCE_SUPPORT / "isolation_admission_forecast.csv"),
        "daily_operations": read_csv(PROCESSED / "daily_operations.csv"),
        "admission_forecast": read_csv(ADMISSION_FORECAST),
        "los_predictions": read_csv(LOS_PREDICTIONS),
        "icu_forecast": read_csv(ICU_FORECAST),
        "occupancy_forecast": read_csv(OCCUPANCY_FORECAST),
        "no_show_predictions": read_csv(NO_SHOW_PREDICTIONS, required=False),
    }
    required_columns = {
        "beds": {"bed_id", "department", "bed_type", "status"},
        "staff": {"staff_id", "department", "role", "availability", "max_hours"},
        "shifts": {"date", "staff_id", "shift", "department", "hours", "assigned"},
        "appointments": {"appointment_id", "doctor_id", "department", "date", "time", "duration"},
        "admissions": {"admission_id", "admission_date", "discharge_date"},
        "isolation_labels": {"admission_id", "isolation_required", "label_source"},
        "isolation_forecast": {
            "date", "department", "predicted_isolation_admissions",
            "historical_smoothed_isolation_rate", "isolation_forecast_source",
        },
        "daily_operations": {"date", "department", "staff_demand"},
        "admission_forecast": {"date", "department", "predicted_admissions"},
        "los_predictions": {"department", "predicted_los_days"},
        "icu_forecast": {"date", "predicted_icu_admissions", "forecast_target_unit"},
        "occupancy_forecast": {"date", "forecast_occupied_beds"},
    }
    for name, columns in required_columns.items():
        missing = columns - set(inputs[name].columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    for name, key in (("beds", "bed_id"), ("staff", "staff_id")):
        snapshot = inputs[name]
        if snapshot[key].duplicated().any():
            raise ValueError(f"{name} support snapshot contains duplicate {key} values.")
        if snapshot["as_of_date"].isna().any():
            raise ValueError(f"{name} support snapshot contains missing as_of_date values.")
    labels = inputs["isolation_labels"]
    if labels["admission_id"].duplicated().any():
        raise ValueError("Isolation label support contains duplicate admission IDs.")
    admissions = inputs["admissions"].merge(
        labels[["admission_id", "isolation_required", "label_source"]],
        on="admission_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (admissions["_merge"] != "both").any():
        raise ValueError("Isolation support labels do not cover every admission.")
    inputs["admissions"] = admissions.drop(columns="_merge")
    admission_forecast = inputs["admission_forecast"].merge(
        inputs["isolation_forecast"][[
            "date", "department", "predicted_isolation_admissions",
            "historical_smoothed_isolation_rate", "isolation_forecast_source",
        ]],
        on=["date", "department"],
        how="left",
        validate="one_to_one",
    )
    if admission_forecast["predicted_isolation_admissions"].isna().any():
        raise ValueError("Isolation forecast does not cover all admission forecast rows.")
    inputs["admission_forecast"] = admission_forecast
    return inputs


def build_bed_forecast(inputs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Project occupied beds using department admissions, LOS, and current census."""
    beds = inputs["beds"].copy()
    beds["_department"] = beds["department"].map(normalize)
    beds["_status"] = beds["status"].map(normalize)
    operational_statuses = {
        "occupied", "in use", "available", "vacant", "open", "ready", "clean", "empty",
    }
    known_statuses = operational_statuses | {"maintenance", "closed", "out of service"}
    unknown_statuses = set(beds["_status"].dropna().unique()) - known_statuses
    if unknown_statuses:
        raise ValueError(f"Unrecognized bed statuses cannot be forecast: {sorted(unknown_statuses)}")
    beds["_operational"] = beds["_status"].isin(operational_statuses)
    beds["_occupied"] = beds["_status"].isin({"occupied", "in use"})
    capacities = beds[beds["_operational"]].groupby("_department").size().to_dict()
    occupancy = beds[beds["_occupied"]].groupby("_department").size().to_dict()
    isolation_occupancy: dict[str, float] = defaultdict(float)

    los = inputs["los_predictions"].copy()
    los["_department"] = los["department"].map(normalize)
    los["_los"] = pd.to_numeric(los["predicted_los_days"], errors="coerce")
    los_by_department = los.groupby("_department")["_los"].mean().to_dict()
    global_los = float(los["_los"].mean())
    if not math.isfinite(global_los) or global_los <= 0:
        raise ValueError("LOS predictions do not contain a positive finite mean.")

    forecast = inputs["admission_forecast"].copy()
    forecast["_date"] = as_date(forecast["date"])
    forecast["_department"] = forecast["department"].map(normalize)
    forecast["_admissions"] = pd.to_numeric(
        forecast["predicted_admissions"], errors="coerce"
    ).fillna(0).clip(lower=0)
    isolation_demand_available = "predicted_isolation_admissions" in forecast.columns
    if isolation_demand_available:
        forecast["_isolation_admissions"] = pd.to_numeric(
            forecast["predicted_isolation_admissions"], errors="coerce"
        ).fillna(0).clip(lower=0)
    forecast = forecast[forecast["_department"] != "all departments"]
    if forecast["_date"].isna().any() or forecast.empty:
        raise ValueError("Admission forecast has no valid department/date rows.")
    aggregation = {"_admissions": "sum"}
    if isolation_demand_available:
        aggregation["_isolation_admissions"] = "sum"
        if "historical_smoothed_isolation_rate" in forecast.columns:
            forecast["_isolation_rate"] = pd.to_numeric(
                forecast["historical_smoothed_isolation_rate"], errors="coerce"
            )
            aggregation["_isolation_rate"] = "mean"
        if "isolation_forecast_source" in forecast.columns:
            forecast["_isolation_source"] = forecast["isolation_forecast_source"]
            aggregation["_isolation_source"] = "first"
    forecast = forecast.groupby(["_date", "_department"], as_index=False).agg(aggregation)
    icu_forecast = inputs.get("icu_forecast", pd.DataFrame()).copy()
    if {"date", "predicted_icu_admissions"} <= set(icu_forecast.columns):
        icu_forecast["_date"] = as_date(icu_forecast["date"])
        icu_forecast["_icu_demand"] = pd.to_numeric(
            icu_forecast["predicted_icu_admissions"], errors="coerce"
        )
        icu_forecast_by_date = icu_forecast.groupby("_date")["_icu_demand"].mean().to_dict()
    else:
        icu_forecast_by_date = {}

    occupancy_benchmark = inputs.get("occupancy_forecast", pd.DataFrame()).copy()
    if {"date", "forecast_occupied_beds"} <= set(occupancy_benchmark.columns):
        occupancy_benchmark["_date"] = as_date(occupancy_benchmark["date"])
        occupancy_benchmark["_occupied"] = pd.to_numeric(
            occupancy_benchmark["forecast_occupied_beds"], errors="coerce"
        )
        occupancy_benchmark = occupancy_benchmark.dropna(subset=["_date", "_occupied"])
        occupancy_by_date = occupancy_benchmark.groupby("_date")["_occupied"].mean().to_dict()
    else:
        occupancy_by_date = {}
    total_operational_beds = int(beds["_operational"].sum())

    rows: list[dict[str, Any]] = []
    for day in sorted(forecast["_date"].unique()):
        daily = forecast[forecast["_date"] == day]
        daily_projection = []
        for department in sorted(set(capacities) | set(occupancy) | set(daily["_department"])):
            admissions = float(
                daily.loc[daily["_department"] == department, "_admissions"].sum()
            )
            previous = float(occupancy.get(department, 0))
            mean_los = los_by_department.get(department, global_los)
            if pd.isna(mean_los) or mean_los <= 0:
                mean_los = global_los
            expected_discharges = min(previous, previous / float(mean_los))
            raw_projected = max(0.0, previous + admissions - expected_discharges)
            admission_los_projected = raw_projected
            icu_model_admissions = icu_forecast_by_date.get(pd.Timestamp(day))
            department_forecast = daily[daily["_department"] == department]
            isolation_rate = (
                department_forecast["_isolation_rate"].iloc[0]
                if "_isolation_rate" in department_forecast.columns
                and not department_forecast.empty
                else pd.NA
            )
            isolation_source = (
                department_forecast["_isolation_source"].iloc[0]
                if "_isolation_source" in department_forecast.columns
                and not department_forecast.empty
                else "missing"
            )
            daily_projection.append({
                "department": department,
                "previous": previous,
                "admissions": admissions,
                "isolation_admissions": (
                    float(daily.loc[
                        daily["_department"] == department,
                        "_isolation_admissions",
                    ].sum())
                    if isolation_demand_available else 0.0
                ),
                "isolation_rate": isolation_rate,
                "isolation_source": isolation_source,
                "isolation_previous": isolation_occupancy.get(department, 0.0),
                "raw_isolation_projected": max(
                    0.0,
                    isolation_occupancy.get(department, 0.0)
                    + (
                        float(daily.loc[
                            daily["_department"] == department,
                            "_isolation_admissions",
                        ].sum())
                        if isolation_demand_available else 0.0
                    )
                    - isolation_occupancy.get(department, 0.0) / float(mean_los),
                ),
                "los": float(mean_los),
                "discharges": expected_discharges,
                "raw_projected": raw_projected,
                "admission_los_projected": admission_los_projected,
                "icu_model_forecast": (
                    icu_model_admissions if department == "icu" else pd.NA
                ),
            })
        raw_total = sum(row["raw_projected"] for row in daily_projection)
        occupancy_target = occupancy_by_date.get(pd.Timestamp(day))
        if occupancy_target is None:
            for item in daily_projection:
                item["projected"] = item["raw_projected"]
                item["calibration_factor"] = 1.0
                item["icu_demand_not_in_occupancy_forecast"] = 0.0
                item["projected_isolation"] = min(
                    item["projected"], item["raw_isolation_projected"]
                )
        else:
            occupancy_target = min(
                float(total_operational_beds), max(0.0, float(occupancy_target))
            )
            icu_projection = sum(
                item["raw_projected"]
                for item in daily_projection if item["department"] == "icu"
            )
            non_icu_projection = sum(
                item["raw_projected"]
                for item in daily_projection if item["department"] != "icu"
            )
            calibrated_icu = min(icu_projection, occupancy_target)
            remaining_occupancy = max(0.0, occupancy_target - calibrated_icu)
            non_icu_factor = (
                remaining_occupancy / non_icu_projection
                if non_icu_projection > 0 else 0.0
            )
            icu_factor = (
                calibrated_icu / icu_projection
                if icu_projection > 0 else 0.0
            )
            for item in daily_projection:
                is_icu = item["department"] == "icu"
                factor = icu_factor if is_icu else non_icu_factor
                item["projected"] = item["raw_projected"] * factor
                item["calibration_factor"] = factor
                item["icu_demand_not_in_occupancy_forecast"] = (
                    max(0.0, icu_projection - calibrated_icu)
                    if is_icu else 0.0
                )
                item["projected_isolation"] = min(
                    item["projected"],
                    item["raw_isolation_projected"] * factor,
                )
        for item in daily_projection:
            department = item["department"]
            projected = item["projected"]
            capacity = int(capacities.get(department, 0))
            rows.append({
                "date": pd.Timestamp(day).date().isoformat(),
                "department": department,
                "current_occupied": round(item["previous"], 2),
                "predicted_admissions": round(item["admissions"], 2),
                "predicted_isolation_admissions": round(item["isolation_admissions"], 2),
                "projected_isolation_bed_demand": round(
                    item["projected_isolation"], 4
                ),
                "isolation_scenario_rate": (
                    round(float(item["isolation_rate"]), 6)
                    if pd.notna(item["isolation_rate"])
                    else pd.NA
                ),
                "isolation_source_type": (
                    str(item["isolation_source"])
                    if pd.notna(item["isolation_source"])
                    else "missing"
                ),
                "isolation_demand_source_available": isolation_demand_available,
                "mean_predicted_los_days": round(item["los"], 2),
                "expected_discharges": round(item["discharges"], 2),
                "unconstrained_projected_occupied_beds": round(item["raw_projected"], 2),
                "admission_los_projected_occupied_beds": round(
                    item["admission_los_projected"], 2
                ),
                "predicted_icu_admissions_per_day": item["icu_model_forecast"],
                "occupancy_forecast_total_beds": (
                    round(float(occupancy_target), 4)
                    if occupancy_target is not None else pd.NA
                ),
                "occupancy_calibration_factor": round(item["calibration_factor"], 4),
                "icu_demand_not_in_occupancy_forecast": round(
                    item["icu_demand_not_in_occupancy_forecast"], 2
                ),
                "projected_occupied_beds": round(projected, 4),
                "operational_bed_capacity": capacity,
                "forecast_available_beds": round(max(0.0, capacity - projected), 4),
            })
            occupancy[department] = projected
            isolation_occupancy[department] = item["projected_isolation"]
    return pd.DataFrame(rows)


def _shift_interval(day: date, shift_name: str, hours: float) -> tuple[datetime, datetime]:
    """Return actual shift boundaries, including overnight shifts."""
    start_minute, _ = SHIFTS[shift_name]
    start = datetime.combine(day, datetime.min.time()) + timedelta(minutes=start_minute)
    return start, start + timedelta(hours=hours)


def optimize_staff_schedule(
    inputs: dict[str, pd.DataFrame],
    plan_date: date,
    planning_dates: list[date] | None = None,
    bed_forecast: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Schedule a horizon using historical role coverage, workload, rest, and hour limits."""
    staff = inputs["staff"].copy().fillna("")
    shifts = inputs["shifts"].copy()
    staff["_department"] = staff["department"].map(normalize)
    staff["_specialty"] = staff["specialty"].map(normalize) if "specialty" in staff else ""
    staff["_role"] = staff["role"].map(normalize)
    staff["_available"] = staff["availability"].map(normalize).isin({"available", "on call"})
    staff["_clinical"] = staff["_role"].map(
        lambda role: any(token in role for token in CLINICAL_ROLE_TOKENS)
    )
    staff["_max_hours"] = pd.to_numeric(staff["max_hours"], errors="coerce").fillna(0)
    shifts["_date"] = as_date(shifts["date"])
    shifts["_assigned"] = shifts["assigned"].map(
        normalize
    ).isin({"yes", "true", "1", "assigned"})
    shifts["_hours"] = pd.to_numeric(shifts["hours"], errors="coerce").fillna(0)
    shifts["_department"] = shifts["department"].map(normalize)
    shifts["_shift"] = shifts["shift"].map(lambda value: str(value).strip().title())
    shifts["_role"] = shifts["staff_role"].map(normalize) if "staff_role" in shifts else ""
    first_date = pd.Timestamp(plan_date)
    history_start = first_date - pd.Timedelta(days=90)
    history = shifts[
        shifts["_assigned"]
        & (shifts["_date"] < first_date)
        & (shifts["_date"] >= history_start)
        & shifts["_shift"].isin(SHIFTS)
    ].copy()
    all_history_days = pd.date_range(history_start, first_date - pd.Timedelta(days=1), freq="D")
    if len(all_history_days) == 0:
        raise ValueError("Staff schedule needs historical dates before planning starts.")

    cell_counts = history.groupby(
        ["_department", "_shift", "_role", "_date"]
    ).size()
    historical_cells = cell_counts.groupby(level=[0, 1, 2]).sum() / len(all_history_days)

    operations = inputs["daily_operations"].copy()
    operations["_date"] = as_date(operations["date"])
    operations["_department"] = operations["department"].map(normalize)
    operations["_admissions"] = pd.to_numeric(
        operations["admissions"], errors="coerce"
    ).fillna(0).clip(lower=0)
    recent_operations = operations[
        (operations["_date"] < first_date)
        & (operations["_date"] >= history_start)
    ]
    historical_admissions = recent_operations.groupby("_department")["_admissions"].mean().to_dict()
    forecast = inputs.get("admission_forecast", pd.DataFrame()).copy()
    if not forecast.empty and {"date", "department", "predicted_admissions"} <= set(forecast.columns):
        forecast["_date"] = as_date(forecast["date"])
        forecast["_department"] = forecast["department"].map(normalize)
        forecast["_admissions"] = pd.to_numeric(
            forecast["predicted_admissions"], errors="coerce"
        ).fillna(0).clip(lower=0)
        forecast = forecast[forecast["_department"] != "all departments"]
        forecast = forecast.groupby(["_date", "_department"], as_index=False)["_admissions"].sum()
    else:
        forecast = pd.DataFrame(columns=["_date", "_department", "_admissions"])

    if planning_dates is None:
        planning_dates = [plan_date]
    planning_dates = sorted(set(planning_dates))
    if not planning_dates or planning_dates[0] < plan_date:
        raise ValueError("Planning dates must start on or after the planning date.")

    duration_by_role_shift = history.groupby(["_role", "_shift"])["_hours"].median().to_dict()
    duration_by_shift = history.groupby("_shift")["_hours"].median().to_dict()
    coverage_by_date: dict[date, dict[tuple[str, str, str], int]] = {}
    maximum_requirements: dict[tuple[str, str, str], int] = defaultdict(int)
    bed_forecast_by_date = pd.DataFrame()
    if bed_forecast is not None and not bed_forecast.empty:
        bed_forecast_by_date = bed_forecast.copy()
        bed_forecast_by_date["_date"] = as_date(bed_forecast_by_date["date"])
        bed_forecast_by_date["_department"] = bed_forecast_by_date["department"].map(normalize)
        bed_forecast_by_date["_demand"] = pd.to_numeric(
            bed_forecast_by_date["projected_occupied_beds"], errors="coerce"
        ).fillna(0).clip(lower=0)

    for target_day in planning_dates:
        day_stamp = pd.Timestamp(target_day)
        forecast_day = forecast[forecast["_date"] == day_stamp]
        forecast_admissions = forecast_day.set_index("_department")["_admissions"].to_dict()
        coverage: dict[tuple[str, str, str], int] = {}
        for (department, shift_name, role), daily_mean in historical_cells.items():
            if daily_mean <= 0:
                continue
            baseline_admissions = float(historical_admissions.get(department, 0))
            expected_admissions = float(forecast_admissions.get(department, baseline_admissions))
            volume_ratio = (
                expected_admissions / baseline_admissions
                if baseline_admissions > 0 else 1.0
            )
            volume_ratio = min(1.5, max(0.5, volume_ratio))
            scaled_demand = float(daily_mean) * volume_ratio
            required = (
                max(1, int(math.floor(scaled_demand + 0.5)))
                if "nurse" in role
                else int(math.floor(scaled_demand + 0.5))
            )
            if required > 0:
                coverage[(department, shift_name, role)] = required

        if not bed_forecast_by_date.empty:
            day_beds = bed_forecast_by_date[bed_forecast_by_date["_date"] == day_stamp]
            for _, bed_row in day_beds.iterrows():
                department = bed_row["_department"]
                nurses_per_shift = int(math.ceil(float(bed_row["_demand"]) / NURSE_TO_BED_CAPACITY))
                for shift_name in SHIFTS:
                    key = (department, shift_name, "nurse")
                    coverage[key] = max(coverage.get(key, 0), nurses_per_shift)

        coverage_by_date[target_day] = coverage
        for key, required in coverage.items():
            maximum_requirements[key] = max(maximum_requirements[key], required)

    staff_rows = {str(row["staff_id"]): row for _, row in staff.iterrows()}
    surge_staff_plan: list[dict[str, Any]] = []
    if not bed_forecast_by_date.empty:
        for (department, shift_name, role), required in sorted(maximum_requirements.items()):
            if required <= 0:
                continue
            shift_hours = float(
                duration_by_role_shift.get(
                    (role, shift_name),
                    duration_by_shift.get(shift_name, SHIFT_HOURS),
                )
            )
            roster_size = max(
                required,
                int(math.ceil(required * shift_hours * 7 / 40)) + 2,
            )
            role_label = " ".join(part.capitalize() for part in role.split())
            rate = (
                75.0 if "nurse" in role
                else 160.0 if any(token in role for token in ("doctor", "physician", "surgeon"))
                else 60.0 if any(token in role for token in CLINICAL_ROLE_TOKENS)
                else 30.0
            )
            for index in range(roster_size):
                staff_id = (
                    f"SURGE-{department.replace(' ', '-')}-{shift_name.upper()}-"
                    f"{role.replace(' ', '-')}-{index + 1:03d}"
                )
                person = pd.Series({
                    "staff_id": staff_id,
                    "staff_name": f"Synthetic surge {role_label} {index + 1}",
                    "role": role_label,
                    "specialty": department,
                    "department": department,
                    "availability": "Available",
                    "shift_preference": shift_name,
                    "max_hours": 40,
                    "hourly_rate_usd": rate,
                    "staff_source": "synthetic_surge_pool",
                    "scenario_only": True,
                    "_department": department,
                    "_specialty": department,
                    "_role": role,
                    "_available": True,
                    "_clinical": any(token in role for token in CLINICAL_ROLE_TOKENS),
                    "_max_hours": 40,
                })
                staff_rows[staff_id] = person
            surge_staff_plan.append({
                "department": department,
                "role": role_label,
                "shift": shift_name,
                "peak_required_staff": required,
                "synthetic_surge_pool_staff": roster_size,
                "hourly_rate_usd": rate,
                "scenario_only": True,
            })
    inputs["synthetic_surge_staff_plan"] = pd.DataFrame(surge_staff_plan)

    schedule_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for target_day in planning_dates:
        day_stamp = pd.Timestamp(target_day)
        coverage = coverage_by_date[target_day]

        prior_start = pd.Timestamp(target_day) - pd.Timedelta(days=6)
        prior_history = shifts[
            shifts["_assigned"]
            & (shifts["_date"] < day_stamp)
            & (shifts["_date"] >= prior_start)
        ]
        planned_prior = [
            row for row in schedule_rows
            if prior_start.date() <= date.fromisoformat(row["date"]) < target_day
        ]
        hours_by_staff: dict[str, float] = defaultdict(float)
        intervals_by_staff: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
        for _, row in prior_history.iterrows():
            staff_id = str(row["staff_id"])
            shift_name = str(row["_shift"])
            if shift_name not in SHIFTS or float(row["_hours"]) <= 0:
                continue
            shift_day = row["_date"].date()
            start, end = _shift_interval(shift_day, shift_name, float(row["_hours"]))
            hours_by_staff[staff_id] += float(row["_hours"])
            intervals_by_staff[staff_id].append((start, end))
        for row in planned_prior:
            staff_id = row["staff_id"]
            start, end = _shift_interval(
                date.fromisoformat(row["date"]), row["shift"], float(row["hours"])
            )
            hours_by_staff[staff_id] += float(row["hours"])
            intervals_by_staff[staff_id].append((start, end))

        model = cp_model.CpModel()
        assignments: dict[tuple[str, str, str, str, int], cp_model.IntVar] = {}
        unfilled: dict[tuple[str, str, str, int], cp_model.IntVar] = {}
        for (department, shift_name, role), required in sorted(coverage.items()):
            hours = float(
                duration_by_role_shift.get(
                    (role, shift_name),
                    duration_by_shift.get(shift_name, SHIFT_HOURS),
                )
            )
            if hours <= 0:
                raise ValueError(f"Invalid historical shift duration for {role}/{shift_name}.")
            start, end = _shift_interval(target_day, shift_name, hours)
            for slot in range(required):
                candidates = []
                for staff_id, person in staff_rows.items():
                    if not person["_available"]:
                        continue
                    if person["_role"] != role:
                        continue
                    if department not in {person["_department"], person["_specialty"]}:
                        continue
                    if hours_by_staff[staff_id] + hours > float(person["_max_hours"]):
                        continue
                    conflicts_with_rest = False
                    for previous_start, previous_end in intervals_by_staff[staff_id]:
                        gap = max(
                            (start - previous_end).total_seconds(),
                            (previous_start - end).total_seconds(),
                        )
                        if gap < 11 * 60 * 60:
                            conflicts_with_rest = True
                            break
                    if conflicts_with_rest:
                        continue
                    key = (staff_id, department, shift_name, role, slot)
                    if key not in assignments:
                        assignments[key] = model.new_bool_var(
                            f"assign_{staff_id}_{department.replace(' ', '_')}_{shift_name}_{role}_{slot}"
                        )
                    variable = assignments[key]
                    candidates.append(variable)
                model.add(sum(candidates) <= 1)
                missing = model.new_bool_var(
                    f"unfilled_{department}_{shift_name}_{role}_{slot}"
                )
                model.add(sum(candidates) + missing == 1)
                unfilled[(department, shift_name, role, slot)] = missing

        for staff_id in staff_rows:
            daily_assignments = [
                variable for (candidate_id, _, _, _, _), variable in assignments.items()
                if candidate_id == staff_id
            ]
            if daily_assignments:
                model.add_at_most_one(daily_assignments)

        objective_terms = []
        for (staff_id, department, shift_name, role, _), variable in assignments.items():
            person = staff_rows[staff_id]
            preference = normalize(person.get("shift_preference", ""))
            preference_bonus = 20 if preference == normalize(shift_name) else 0
            specialty_bonus = 10 if person["_specialty"] == department else 0
            workload_penalty = int(hours_by_staff[staff_id] * 2)
            department_priority = (
                2_000 if department == "emergency"
                else 1_000 if department == "icu"
                else 0
            )
            objective_terms.append(
                (10_000 + department_priority + preference_bonus
                 + specialty_bonus - workload_penalty) * variable
            )
        missing_penalties = []
        for (department, _, _, _), missing in unfilled.items():
            penalty = (
                200_000 if department == "emergency"
                else 150_000 if department == "icu"
                else 100_000
            )
            missing_penalties.append(penalty * missing)
        model.maximize(sum(objective_terms) - sum(missing_penalties))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = SOLVER_SECONDS
        solver.parameters.num_search_workers = SOLVER_WORKERS
        solver.parameters.random_seed = SOLVER_RANDOM_SEED
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(
                f"Staff scheduling solver failed on {target_day}: {solver.status_name(status)}"
            )

        todays_schedule_rows = []
        for (staff_id, department, shift_name, role, slot), variable in assignments.items():
            if not solver.value(variable):
                continue
            person = staff_rows[staff_id]
            hours = float(
                duration_by_role_shift.get(
                    (role, shift_name),
                    duration_by_shift.get(shift_name, SHIFT_HOURS),
                )
            )
            scheduled_row = {
                "date": target_day.isoformat(),
                "staff_id": staff_id,
                "staff_name": person.get("staff_name", ""),
                "role": person["role"],
                "specialty": person.get("specialty", ""),
                "department": department,
                "shift": shift_name,
                "hours": hours,
                "max_hours": int(person["_max_hours"]),
                "prior_7_day_hours": hours_by_staff[staff_id],
                "shift_preference": person.get("shift_preference", ""),
                "coverage_slot": slot,
                "staff_source": person.get("staff_source", "base_synthetic_workforce"),
                "hourly_rate_usd": person.get("hourly_rate_usd", pd.NA),
                "scenario_only": bool(person.get("scenario_only", False)),
            }
            schedule_rows.append(scheduled_row)
            todays_schedule_rows.append(scheduled_row)

        for (department, shift_name, role), required in sorted(coverage.items()):
            assigned_count = int(
                (
                    (pd.DataFrame(todays_schedule_rows)["department"] == department)
                    & (pd.DataFrame(todays_schedule_rows)["shift"] == shift_name)
                    & (pd.DataFrame(todays_schedule_rows)["role"].map(normalize) == role)
                ).sum()
            ) if todays_schedule_rows else 0
            coverage_rows.append({
                "date": target_day.isoformat(),
                "department": department,
                "shift": shift_name,
                "role": role,
                "required_staff": required,
                "assigned_staff": assigned_count,
                "unfilled_staff": required - assigned_count,
            })

    return pd.DataFrame(schedule_rows), pd.DataFrame(coverage_rows)


def optimize_appointments(inputs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-slot the latest appointment day, preserving doctor, duration, and urgency."""
    appointments = inputs["appointments"].copy()
    appointments["_date"] = as_date(appointments["date"])
    appointments = appointments[appointments["_date"].notna()]
    if appointments.empty:
        raise ValueError("No appointments with valid dates are available.")
    plan_day = appointments["_date"].max()
    appointments = appointments[appointments["_date"] == plan_day].copy()
    appointments["_plan_day"] = plan_day
    appointments = appointments.sort_values(
        ["_plan_day", "appointment_id"]
    ).reset_index(drop=True)
    appointments["duration"] = pd.to_numeric(appointments["duration"], errors="coerce")
    if appointments["duration"].isna().any() or (appointments["duration"] <= 0).any():
        raise ValueError("Appointment durations must be positive numeric minutes.")
    appointments["duration"] = appointments["duration"].clip(lower=15).map(
        lambda value: int(math.ceil(value / 15) * 15)
    )

    risks = inputs["no_show_predictions"]
    if not risks.empty and {"appointment_id", "predicted_no_show_probability"} <= set(risks.columns):
        risks = risks[["appointment_id", "predicted_no_show_probability"]].drop_duplicates("appointment_id")
        appointments = appointments.merge(risks, on="appointment_id", how="left")
    else:
        appointments["predicted_no_show_probability"] = pd.NA
    history = inputs["appointments"].copy()
    history["_date"] = as_date(history["date"])
    if "no_show" in history.columns:
        history["_no_show"] = pd.to_numeric(history["no_show"], errors="coerce")
        history = history[
            (history["_date"] < plan_day)
            & history["_no_show"].isin([0, 1])
        ]
    else:
        history = pd.DataFrame(columns=["department", "_no_show"])
    fallback_rate = (
        float(history["_no_show"].mean()) if not history.empty else 0.0
    )
    department_priors = {}
    if not history.empty:
        department_stats = history.assign(
            _department=history["department"].map(normalize)
        ).groupby("_department")["_no_show"].agg(["sum", "count"])
        department_priors = {
            department: (float(row["sum"]) + 20 * fallback_rate)
            / (int(row["count"]) + 20)
            for department, row in department_stats.iterrows()
        }
    historical_fallback = appointments["department"].map(
        lambda department: department_priors.get(normalize(department), fallback_rate)
    )
    appointments["no_show_risk"] = pd.to_numeric(
        appointments["predicted_no_show_probability"], errors="coerce"
    ).fillna(historical_fallback).clip(0, 1)
    department_appointment_counts = appointments["department"].map(
        normalize
    ).value_counts().to_dict()

    staff = inputs["staff"].copy().fillna("")
    staff["_department"] = staff["department"].map(normalize)
    staff["_specialty"] = staff.get("specialty", "").map(normalize) if "specialty" in staff else ""
    staff["_role"] = staff["role"].map(normalize)
    staff["_available"] = staff["availability"].map(normalize).isin({"available", "on call"})
    staff["_doctor"] = staff["_role"].map(
        lambda role: any(token in role for token in ("doctor", "physician", "surgeon", "practitioner"))
    )
    providers = {
        str(row["staff_id"]): row
        for _, row in staff[staff["_available"] & staff["_doctor"]].iterrows()
    }
    provider_windows: dict[str, list[tuple[datetime, datetime]]] = {}
    provider_shifts = inputs.get("shifts", pd.DataFrame()).copy()
    has_roster_for_day = False
    if (
        not provider_shifts.empty
        and {"date", "staff_id", "shift", "assigned"} <= set(provider_shifts.columns)
    ):
        provider_shifts["_date"] = as_date(provider_shifts["date"])
        provider_shifts["_assigned"] = provider_shifts["assigned"].map(
            normalize
        ).isin({"yes", "true", "1", "assigned"})
        if "hours" in provider_shifts.columns:
            provider_shifts["_hours"] = pd.to_numeric(
                provider_shifts["hours"], errors="coerce"
            ).fillna(SHIFT_HOURS)
        else:
            provider_shifts["_hours"] = SHIFT_HOURS
        on_day = provider_shifts[
            (provider_shifts["_date"] == plan_day)
            & provider_shifts["_assigned"]
            & provider_shifts["staff_id"].astype(str).isin(providers)
        ]
        has_roster_for_day = bool(
            (
                (provider_shifts["_date"] == plan_day)
                & provider_shifts["_assigned"]
            ).any()
        )
        for provider_id, group in on_day.groupby(on_day["staff_id"].astype(str)):
            windows = []
            for _, shift_row in group.iterrows():
                shift_name = str(shift_row["shift"]).strip().title()
                if shift_name not in SHIFTS:
                    raise ValueError(
                        f"Unknown provider shift {shift_row['shift']!r} for {provider_id}."
                    )
                windows.append(
                    _shift_interval(plan_day.date(), shift_name, float(shift_row["_hours"]))
                )
            provider_windows[provider_id] = windows

    model = cp_model.CpModel()
    selected: dict[tuple[int, str, datetime], cp_model.IntVar] = {}
    optional_intervals: dict[str, list[cp_model.IntervalVar]] = defaultdict(list)
    unbooked: dict[int, cp_model.IntVar] = {}
    original_minutes: list[int] = []
    appointment_datetimes: list[datetime] = []
    for index, row in appointments.iterrows():
        try:
            original = datetime.strptime(
                f"{row['_plan_day'].date().isoformat()} {str(row['time'])[:5]}",
                "%Y-%m-%d %H:%M",
            )
            original_minute = original.hour * 60 + original.minute
        except ValueError as exc:
            raise ValueError(f"Invalid appointment time for {row['appointment_id']}: {row['time']}") from exc
        original_minutes.append(original_minute)
        appointment_datetimes.append(original)
        duration = int(row["duration"])
        day_start = datetime.combine(row["_plan_day"].date(), datetime.min.time())
        candidates = [
            day_start + timedelta(minutes=minute)
            for minute in range(8 * 60, 17 * 60 - duration + 1, 15)
        ]
        department = normalize(row["department"])
        eligible_providers = [
            provider_id for provider_id, person in providers.items()
            if department in {person["_department"], person["_specialty"]}
        ]
        if has_roster_for_day:
            appointment_providers = [
                provider_id for provider_id in eligible_providers
                if provider_id in provider_windows
            ]
        else:
            appointment_providers = eligible_providers
        provider_option_limit = min(
            12,
            max(
                3,
                math.ceil(department_appointment_counts.get(department, 0) / 6),
            ),
        )
        day_end = day_start + timedelta(hours=17)
        original_provider = str(row["doctor_id"])
        appointment_providers = sorted(
            appointment_providers,
            key=lambda provider_id: (
                sum(
                    max(
                        0.0,
                        (
                            min(window_end, day_end)
                            - max(window_start, day_start)
                        ).total_seconds() / 60,
                    )
                    for window_start, window_end in provider_windows.get(
                        provider_id, [(day_start, day_end)]
                    )
                ),
                provider_id == original_provider,
            ),
            reverse=True,
        )[:provider_option_limit]
        options = []
        for provider_id in appointment_providers:
            windows = provider_windows.get(provider_id)
            for candidate in candidates:
                if windows is not None:
                    appointment_end = candidate + timedelta(minutes=duration)
                    if not any(
                        window_start <= candidate
                        and appointment_end <= window_end
                        for window_start, window_end in windows
                    ):
                        continue
                variable = model.new_bool_var(
                    f"appt_{index}_{provider_id}_{candidate.strftime('%Y%m%d%H%M')}"
                )
                selected[(index, provider_id, candidate)] = variable
                options.append(variable)
                interval = model.new_optional_fixed_size_interval_var(
                    int(candidate.timestamp() // 60), duration, variable,
                    f"appt_interval_{index}_{provider_id}_{candidate.strftime('%Y%m%d%H%M')}",
                )
                optional_intervals[provider_id].append(interval)
        unbooked[index] = model.new_bool_var(f"appt_unbooked_{index}")
        model.add(sum(options) + unbooked[index] == 1)
    for intervals in optional_intervals.values():
        model.add_no_overlap(intervals)

    objective = []
    for (index, provider_id, start), variable in selected.items():
        row = appointments.iloc[index]
        priority = normalize(row.get("priority", "routine"))
        priority_weight = 3 if priority in {"emergency", "critical"} else 2 if priority in {"urgent", "high"} else 1
        severity = normalize(row.get("patient_severity", ""))
        severity_weight = 2 if severity in {"critical", "high", "severe"} else 1
        movement_cost = 4
        distance = int(abs((start - appointment_datetimes[index]).total_seconds()) // 60)
        start_minute = start.hour * 60 + start.minute
        provider_change_cost = 0 if provider_id == str(row["doctor_id"]) else 50_000
        objective.append(
            (1_000_000 + priority_weight * 10_000 + severity_weight * 1_000
             - distance * movement_cost - start_minute // 15 - provider_change_cost) * variable
        )
    model.maximize(sum(objective) - sum(unbooked.values()) * 10_000_000)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_SECONDS
    solver.parameters.num_search_workers = SOLVER_WORKERS
    solver.parameters.random_seed = SOLVER_RANDOM_SEED
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Appointment scheduling solver failed: {solver.status_name(status)}")

    output = []
    for index, row in appointments.iterrows():
        assignments = [
            (provider_id, start)
            for (candidate, provider_id, start), var in selected.items()
            if candidate == index and solver.value(var)
        ]
        provider_id, assigned_start = assignments[0] if assignments else ("", None)
        original_datetime = appointment_datetimes[index]
        output.append({
            "appointment_id": row["appointment_id"],
            "patient_id": row.get("patient_id", ""),
            "original_doctor_id": row["doctor_id"],
            "optimized_doctor_id": provider_id,
            "optimized_doctor_name": (
                providers[provider_id].get("staff_name", "") if provider_id else ""
            ),
            "department": row["department"],
            "date": row["_plan_day"].date().isoformat(),
            "original_time": str(row["time"]),
            "optimized_time": (
                assigned_start.strftime("%H:%M")
                if assigned_start is not None else ""
            ),
            "duration_minutes": int(row["duration"]),
            "priority": row.get("priority", ""),
            "no_show_risk": round(float(row["no_show_risk"]), 4),
            "status": "SCHEDULED" if assigned_start is not None else "UNSCHEDULED",
        })
    result = pd.DataFrame(output)
    appointments["plan_date"] = appointments["_plan_day"].dt.strftime("%Y-%m-%d")
    result["plan_date"] = result["date"]
    before_conflicts = count_appointment_conflicts(
        appointments, "time", "doctor_id", "plan_date"
    )
    after_conflicts = count_appointment_conflicts(
        result, "optimized_time", "optimized_doctor_id", "plan_date"
    )
    availability_by_provider = {
        str(row["staff_id"]): bool(row["_available"])
        for _, row in staff.iterrows()
    }
    staff_by_id = {str(row["staff_id"]): row for _, row in staff.iterrows()}
    baseline_specialty_mismatches = 0
    for _, row in appointments.iterrows():
        person = staff_by_id.get(str(row["doctor_id"]))
        department = normalize(row["department"])
        if person is None or department not in {
            person["_department"], person["_specialty"]
        }:
            baseline_specialty_mismatches += 1
    metrics = pd.DataFrame([{
        "planning_date_start": appointments["plan_date"].min(),
        "planning_date_end": appointments["plan_date"].max(),
        "appointments_requested": len(result),
        "appointments_scheduled": int((result["status"] == "SCHEDULED").sum()),
        "appointments_unassigned": int((result["status"] != "SCHEDULED").sum()),
        "baseline_unavailable_doctor_bookings": sum(
            not availability_by_provider.get(str(doctor_id), False)
            for doctor_id in appointments["doctor_id"]
        ),
        "baseline_off_shift_doctor_bookings": (
            sum(
                str(row["doctor_id"]) not in provider_windows
                for _, row in appointments.iterrows()
            )
            if has_roster_for_day else 0
        ),
        "baseline_specialty_mismatch_bookings": baseline_specialty_mismatches,
        "optimized_unavailable_doctor_bookings": sum(
            not availability_by_provider.get(str(doctor_id), False)
            for doctor_id in result.loc[
                result["status"] == "SCHEDULED", "optimized_doctor_id"
            ]
        ),
        "optimized_off_shift_doctor_bookings": (
            sum(
                str(doctor_id) not in provider_windows
                for doctor_id in result.loc[
                    result["status"] == "SCHEDULED", "optimized_doctor_id"
                ]
            )
            if has_roster_for_day else 0
        ),
        "baseline_doctor_time_conflicts": before_conflicts,
        "optimized_doctor_time_conflicts": after_conflicts,
        "doctor_changes": int(
            (result["original_doctor_id"] != result["optimized_doctor_id"]).sum()
        ),
        "mean_no_show_risk": round(float(result["no_show_risk"].mean()), 4),
        "mean_schedule_shift_minutes": round(
            float((result.apply(
                lambda row: abs(
                        (
                            datetime.strptime(
                                f"{row['date']} {row['optimized_time']}",
                                "%Y-%m-%d %H:%M",
                            )
                            - datetime.strptime(
                                f"{row['date']} {str(row['original_time'])[:5]}",
                                "%Y-%m-%d %H:%M",
                            )
                        ).total_seconds() // 60
                    ) if row["optimized_time"] else 0,
                axis=1,
            )).mean()), 2
        ),
    }])
    return result, metrics


def count_appointment_conflicts(
    frame: pd.DataFrame,
    time_column: str,
    doctor_column: str = "doctor_id",
    date_column: str | None = None,
) -> int:
    """Count overlapping appointments for the same doctor."""
    conflicts = 0
    if frame.empty:
        return conflicts
    grouping = [doctor_column]
    if date_column and date_column in frame.columns:
        grouping.append(date_column)
    for _, group in frame.groupby(grouping):
        intervals = []
        for _, row in group.iterrows():
            try:
                time_string = str(row[time_column])[:5]
                if date_column and date_column in group.columns:
                    start = datetime.strptime(
                        f"{row[date_column]} {time_string}", "%Y-%m-%d %H:%M"
                    )
                else:
                    start = datetime.strptime(time_string, "%H:%M")
            except ValueError:
                continue
            duration = row.get("duration", row.get("duration_minutes"))
            if pd.isna(duration):
                continue
            end = start + timedelta(minutes=int(duration))
            intervals.append((start, end))
        intervals.sort()
        latest_end: datetime | None = None
        for start, end in intervals:
            if latest_end is not None and start < latest_end:
                conflicts += 1
            if latest_end is None or end > latest_end:
                latest_end = end
    return conflicts


def optimize_bed_allocations(
    forecast: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    staff_coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare a no-transfer baseline with safe CP-SAT bed pooling and transfers."""
    beds = inputs["beds"].copy()
    beds["_department"] = beds["department"].map(normalize)
    beds["_bed_type"] = beds["bed_type"].map(normalize)
    beds["_status"] = beds["status"].map(normalize)
    beds["_isolation"] = (
        beds["isolation_capable"].map(is_true)
        if "isolation_capable" in beds.columns else False
    )
    beds["_emergency_eligible"] = (
        beds["available_for_emergency"].map(is_true)
        if "available_for_emergency" in beds.columns else False
    )
    operational_statuses = {
        "occupied", "in use", "available", "vacant", "open", "ready", "clean", "empty",
    }
    unknown_statuses = set(beds["_status"].dropna().unique()) - operational_statuses - {
        "maintenance", "closed", "out of service",
    }
    if unknown_statuses:
        raise ValueError(f"Unrecognized bed statuses cannot be assumed usable: {sorted(unknown_statuses)}")
    beds = beds[beds["_status"].isin(operational_statuses)].copy()
    pool_columns = ["_department", "_bed_type", "_isolation", "_emergency_eligible"]
    pool_capacities = beds.groupby(pool_columns, dropna=False).size().to_dict()
    pools = sorted(pool_capacities)
    type_capacity: dict[str, int] = defaultdict(int)
    for pool, capacity in pool_capacities.items():
        type_capacity[pool[1]] += int(capacity)
    departments = sorted(forecast["department"].unique())

    def is_compatible(department: str, pool: tuple[Any, ...], requires_isolation: bool) -> bool:
        pool_department, bed_type, isolation_capable, emergency_eligible = pool
        if requires_isolation and not isolation_capable:
            return False
        if department == "icu":
            return bed_type == "icu"
        if department == "emergency":
            return bed_type == "emergency" or emergency_eligible
        if bed_type in {"icu", "emergency"}:
            return False
        expected_type = department_bed_type(department)
        return pool_department == department or bed_type == expected_type

    def pool_reserve(pool: tuple[Any, ...]) -> int:
        _, bed_type, _, _ = pool
        capacity = int(pool_capacities[pool])
        if bed_type == "icu":
            return min(ICU_RESERVE, capacity)
        if bed_type == "emergency":
            return min(EMERGENCY_RESERVE, capacity)
        return 0

    def round_daily_census(daily: pd.DataFrame) -> dict[str, int]:
        raw = {
            str(row["department"]): max(0.0, float(row["projected_occupied_beds"]))
            for _, row in daily.iterrows()
        }
        rounded_total = int(math.floor(sum(raw.values()) + 0.5))
        counts = {department: int(math.floor(value)) for department, value in raw.items()}
        remainder = rounded_total - sum(counts.values())
        order = sorted(raw, key=lambda department: raw[department] - counts[department], reverse=True)
        for department in order[:remainder]:
            counts[department] += 1
        return counts

    def staffing_capacity_for_day(day: str) -> dict[str, int]:
        return {
            department: staffed_bed_capacity(staff_coverage, department, day)
            for department in departments
        }

    results = []
    for day in sorted(forecast["date"].astype(str).unique()):
        daily = forecast[forecast["date"].astype(str) == day]
        demand = round_daily_census(daily)
        isolation_known = (
            "predicted_isolation_admissions" in daily.columns
            and bool(daily["isolation_demand_source_available"].map(is_true).any())
        )
        isolation_demand: dict[str, int] = {}
        for department in departments:
            if isolation_known and "projected_isolation_bed_demand" in daily.columns:
                requested_isolation = int(math.ceil(pd.to_numeric(
                    daily.loc[
                        daily["department"] == department,
                        "projected_isolation_bed_demand",
                    ],
                    errors="coerce",
                ).fillna(0).sum()))
            elif isolation_known:
                requested_isolation = int(math.ceil(pd.to_numeric(
                    daily.loc[
                        daily["department"] == department,
                        "predicted_isolation_admissions",
                    ],
                    errors="coerce",
                ).fillna(0).sum()))
            else:
                requested_isolation = 0
            isolation_demand[department] = min(demand.get(department, 0), requested_isolation)
        staff_caps = staffing_capacity_for_day(day)
        baseline_by_dept: dict[str, int] = defaultdict(int)
        baseline_isolation_by_dept: dict[str, int] = defaultdict(int)
        baseline_pool_used: dict[tuple[Any, ...], int] = defaultdict(int)
        baseline_pool_remaining = {
            pool: max(0, int(pool_capacities[pool]) - pool_reserve(pool))
            for pool in pools
        }
        baseline_staff_remaining = staff_caps.copy()
        for department in sorted(
            departments,
            key=lambda value: (0 if value == "icu" else 1 if value == "emergency" else 2, value),
        ):
            native_pools = [
                pool for pool in pools
                if pool[0] == department and is_compatible(department, pool, False)
            ]
            required_isolation = isolation_demand[department]
            for requires_isolation, target in (
                (True, required_isolation),
                (False, demand.get(department, 0) - required_isolation),
            ):
                remaining = min(target, baseline_staff_remaining.get(department, 0))
                for pool in native_pools:
                    if requires_isolation and not pool[2]:
                        continue
                    safe_capacity = baseline_pool_remaining[pool]
                    if not requires_isolation and pool[1] == "isolation":
                        safe_capacity = max(
                            0, safe_capacity - ISOLATION_RESERVE
                        )
                    assigned = min(remaining, safe_capacity)
                    baseline_by_dept[department] += assigned
                    if requires_isolation:
                        baseline_isolation_by_dept[department] += assigned
                    baseline_pool_used[pool] += assigned
                    baseline_pool_remaining[pool] -= assigned
                    baseline_staff_remaining[department] -= assigned
                    remaining -= assigned
                    if remaining <= 0:
                        break

        model = cp_model.CpModel()
        variables: dict[tuple[str, tuple[Any, ...], bool], cp_model.IntVar] = {}
        for department in departments:
            for pool in pools:
                for requires_isolation in (False, True):
                    if not is_compatible(department, pool, requires_isolation):
                        continue
                    upper_bound = isolation_demand[department] if requires_isolation else demand.get(department, 0)
                    if upper_bound <= 0:
                        continue
                    safe_name = "_".join(str(value).replace(" ", "_") for value in pool)
                    variables[(department, pool, requires_isolation)] = model.new_int_var(
                        0,
                        upper_bound,
                        f"bed_{department.replace(' ', '_')}_{safe_name}_{int(requires_isolation)}",
                    )

        for department in departments:
            dept_variables = [
                value for (dept, _, _), value in variables.items() if dept == department
            ]
            if dept_variables:
                model.add(sum(dept_variables) <= demand.get(department, 0))
                model.add(sum(dept_variables) <= staff_caps.get(department, 0))
                isolation_variables = [
                    value for (dept, _, requires_isolation), value in variables.items()
                    if dept == department and requires_isolation
                ]
                if isolation_variables:
                    model.add(sum(isolation_variables) <= isolation_demand[department])

        for pool in pools:
            pool_variables = [
                value for (_, candidate_pool, _), value in variables.items()
                if candidate_pool == pool
            ]
            if not pool_variables:
                continue
            capacity = max(0, int(pool_capacities[pool]) - pool_reserve(pool))
            model.add(sum(pool_variables) <= capacity)
            if pool[1] == "isolation":
                non_isolation = [
                    value for (_, candidate_pool, requires_isolation), value in variables.items()
                    if candidate_pool == pool and not requires_isolation
                ]
                if non_isolation:
                    model.add(sum(non_isolation) <= max(0, capacity - ISOLATION_RESERVE))

        objective = []
        for (department, pool, requires_isolation), variable in variables.items():
            native = pool[0] == department
            priority = 4 if department == "icu" else 3 if department == "emergency" else 1
            transfer_penalty = 0 if native else 1_000
            isolation_bonus = 2_000 if requires_isolation else 0
            objective.append(
                (1_000_000 + priority * 10_000 + isolation_bonus
                 - transfer_penalty) * variable
            )
        model.maximize(sum(objective))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = SOLVER_SECONDS
        solver.parameters.num_search_workers = SOLVER_WORKERS
        solver.parameters.random_seed = SOLVER_RANDOM_SEED
        status = solver.solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            raise RuntimeError(f"Bed allocation solver failed on {day}: {solver.status_name(status)}")

        optimized_by_dept = defaultdict(int)
        optimized_transfer_in = defaultdict(int)
        optimized_iso = defaultdict(int)
        optimized_pool_use: dict[tuple[Any, ...], int] = defaultdict(int)
        optimized_emergency_flex = 0
        for (department, pool, requires_isolation), variable in variables.items():
            value = int(solver.value(variable))
            optimized_by_dept[department] += value
            optimized_pool_use[pool] += value
            if requires_isolation:
                optimized_iso[department] += value
            if pool[0] != department:
                optimized_transfer_in[department] += value
            if department == "emergency" and (pool[0] != "emergency" or pool[1] != "emergency"):
                optimized_emergency_flex += value

        for department in departments:
            unassigned_isolation = max(
                0, isolation_demand[department] - optimized_iso[department]
            )
            results.append({
                "date": day,
                "department": department,
                "projected_occupied_beds": demand.get(department, 0),
                "projected_isolation_bed_demand": isolation_demand[department],
                "isolation_demand_source_available": isolation_known,
                "baseline_allocated": baseline_by_dept[department],
                "optimized_allocated": optimized_by_dept[department],
                "baseline_overflow": max(0, demand.get(department, 0) - baseline_by_dept[department]),
                "optimized_overflow": max(0, demand.get(department, 0) - optimized_by_dept[department]),
                "baseline_overflow_due_to_staffing": max(
                    0, demand.get(department, 0) - staff_caps.get(department, 0)
                ),
                "optimized_overflow_due_to_staffing": max(
                    0, demand.get(department, 0) - staff_caps.get(department, 0)
                ),
                "baseline_overflow_due_to_bed_constraints": max(
                    0,
                    min(demand.get(department, 0), staff_caps.get(department, 0))
                    - baseline_by_dept[department],
                ),
                "optimized_overflow_due_to_bed_constraints": max(
                    0,
                    min(demand.get(department, 0), staff_caps.get(department, 0))
                    - optimized_by_dept[department],
                ),
                "baseline_unmet_isolation_demand": max(
                    0, isolation_demand[department] - baseline_isolation_by_dept[department]
                ),
                "optimized_unmet_isolation_demand": unassigned_isolation,
                "optimized_transfer_in": optimized_transfer_in[department],
                "staff_bed_capacity": staff_caps.get(department, 0),
                "optimized_emergency_flex_beds": (
                    optimized_emergency_flex if department == "emergency" else 0
                ),
                "baseline_icu_beds_allocated": (
                    baseline_by_dept[department] if department == "icu" else 0
                ),
                "optimized_icu_beds_allocated": (
                    optimized_by_dept[department] if department == "icu" else 0
                ),
                "baseline_emergency_beds_allocated": (
                    baseline_by_dept[department] if department == "emergency" else 0
                ),
                "optimized_emergency_beds_allocated": (
                    optimized_by_dept[department] if department == "emergency" else 0
                ),
                "baseline_icu_reserve_beds_used": (
                    baseline_by_dept[department] if department == "icu" else 0
                ),
                "optimized_icu_reserve_beds_used": (
                    optimized_by_dept[department] if department == "icu" else 0
                ),
                "baseline_emergency_reserve_beds_used": (
                    sum(value for pool, value in baseline_pool_used.items()
                        if pool[1] == "emergency")
                    if department == "emergency" else 0
                ),
                "optimized_emergency_reserve_beds_used": (
                    sum(value for pool, value in optimized_pool_use.items()
                        if pool[1] == "emergency")
                    if department == "emergency" else 0
                ),
                "emergency_eligible_beds_used": (
                    sum(
                        value for pool, value in optimized_pool_use.items()
                        if pool[3] and department == "emergency"
                    )
                    if department == "emergency" else 0
                ),
            })

    detail = pd.DataFrame(results)
    metric_rows = []
    physical_capacity = sum(pool_capacities.values())
    for scenario, allocated_column, overflow_column in (
        ("baseline", "baseline_allocated", "baseline_overflow"),
        ("optimized", "optimized_allocated", "optimized_overflow"),
    ):
        allocated = int(detail[allocated_column].sum())
        overflow = int(detail[overflow_column].sum())
        admissions = float(forecast["predicted_admissions"].sum())
        violations = int((detail[allocated_column] > detail["staff_bed_capacity"]).sum())
        daily_allocations = detail.groupby("date")
        physical_violations = int(
            (daily_allocations[allocated_column].sum() > physical_capacity).sum()
        )
        icu_used_column = f"{scenario}_icu_reserve_beds_used"
        emergency_used_column = f"{scenario}_emergency_reserve_beds_used"
        icu_reserve_capacity = max(
            0, type_capacity.get("icu", 0) - min(ICU_RESERVE, type_capacity.get("icu", 0))
        )
        emergency_reserve_capacity = max(
            0,
            type_capacity.get("emergency", 0)
            - min(EMERGENCY_RESERVE, type_capacity.get("emergency", 0)),
        )
        reserve_violations = int(
            (
                (daily_allocations[icu_used_column].sum() > icu_reserve_capacity)
                | (daily_allocations[emergency_used_column].sum() > emergency_reserve_capacity)
            ).sum()
        )
        daily_forecast = forecast.groupby("date")["projected_occupied_beds"].sum()
        physical_capacity_bed_days = physical_capacity * len(daily_allocations)
        staffed_capacity_bed_days = int(
            detail.groupby("date")["staff_bed_capacity"].sum().sum()
        )
        metric_rows.append({
            "scenario": scenario,
            "forecast_department_days": len(detail),
            "forecast_bed_days": round(float(detail["projected_occupied_beds"].sum()), 2),
            "allocated_bed_days": allocated,
            "overflow_bed_days": overflow,
            "mean_forecast_occupancy_utilization": round(
                float(daily_forecast.mean()) / max(1, physical_capacity), 4
            ),
            "allocated_physical_capacity_utilization": round(
                allocated / max(1, physical_capacity_bed_days), 4
            ),
            "staffed_capacity_utilization": round(
                allocated / max(1, staffed_capacity_bed_days), 4
            ),
            "overflow_bed_days_per_forecast_admission": round(
                overflow / max(1, admissions), 4
            ),
            "overflow_bed_days_due_to_staffing": int(
                detail[f"{scenario}_overflow_due_to_staffing"].sum()
            ),
            "overflow_bed_days_due_to_bed_constraints": int(
                detail[f"{scenario}_overflow_due_to_bed_constraints"].sum()
            ),
            "transfer_in_bed_days": (
                0 if scenario == "baseline"
                else int(detail["optimized_transfer_in"].sum())
            ),
            "unmet_isolation_bed_days": int(
                detail[
                    "baseline_unmet_isolation_demand"
                    if scenario == "baseline"
                    else "optimized_unmet_isolation_demand"
                ].sum()
            ),
            "isolation_demand_source_days_missing": int(
                (~detail["isolation_demand_source_available"]).sum()
            ),
            "staffing_constraint_violations": violations,
            "physical_capacity_violations": physical_violations,
            "emergency_icu_reserve_violations": reserve_violations,
        })
    kpis = pd.DataFrame(metric_rows)
    return detail, kpis


def build_resource_gap_analysis(
    bed_forecast: pd.DataFrame,
    bed_detail: pd.DataFrame,
    staff_coverage: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify daily nurse and dedicated bed gaps without treating estimates as supply."""
    forecast = bed_forecast[[
        "date", "department", "projected_occupied_beds", "operational_bed_capacity",
    ]].copy()
    forecast["department"] = forecast["department"].map(normalize)
    detail_columns = [
        "date", "department", "projected_occupied_beds",
        "optimized_allocated", "optimized_overflow",
        "optimized_overflow_due_to_staffing",
        "optimized_overflow_due_to_bed_constraints",
    ]
    result = forecast.merge(
        bed_detail[detail_columns].assign(
            department=bed_detail["department"].map(normalize)
        ),
        on=["date", "department"],
        how="left",
        validate="one_to_one",
    )
    result["date"] = result["date"].astype(str)
    result["allocation_demand_beds"] = result["projected_occupied_beds_y"]
    result = result.drop(columns=["projected_occupied_beds_y"]).rename(
        columns={"projected_occupied_beds_x": "projected_occupied_beds"}
    )
    result["forecast_occupied_beds"] = result["projected_occupied_beds"]
    result["dedicated_bed_shortfall"] = (
        result["forecast_occupied_beds"] - result["operational_bed_capacity"]
    ).clip(lower=0)

    coverage = staff_coverage.copy()
    coverage["date"] = coverage["date"].astype(str)
    coverage["_department"] = coverage["department"].map(normalize)
    coverage["_role"] = coverage["role"].map(normalize)
    coverage["_shift"] = coverage["shift"].map(
        lambda value: str(value).strip().title()
    )
    nurses = coverage[coverage["_role"] == "nurse"]
    nurse_counts = nurses.pivot_table(
        index=["date", "_department"],
        columns="_shift",
        values="assigned_staff",
        aggfunc="sum",
        fill_value=0,
    )
    for shift in SHIFTS:
        if shift not in nurse_counts.columns:
            nurse_counts[shift] = 0
    nurse_counts["minimum_nurses_per_shift"] = nurse_counts[list(SHIFTS)].min(axis=1)
    nurse_counts = nurse_counts[["minimum_nurses_per_shift"]].reset_index()
    result = result.merge(
        nurse_counts,
        left_on=["date", "department"],
        right_on=["date", "_department"],
        how="left",
        validate="one_to_one",
    ).drop(columns="_department")
    result["minimum_nurses_per_shift"] = (
        result["minimum_nurses_per_shift"].fillna(0).astype(int)
    )
    result["nurses_required_per_shift"] = (
        result["forecast_occupied_beds"] / NURSE_TO_BED_CAPACITY
    ).map(math.ceil).astype(int)
    result["incremental_nurses_required_per_shift"] = (
        result["nurses_required_per_shift"] - result["minimum_nurses_per_shift"]
    ).clip(lower=0)
    return result


def optimize_emergency_capacity(
    forecast: pd.DataFrame,
    inputs: dict[str, pd.DataFrame],
    staff_coverage: pd.DataFrame,
    bed_allocation_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Report ICU/ED expansion capacity from the shared bed-allocation solution."""
    beds = inputs["beds"].copy()
    beds["_type"] = beds["bed_type"].map(normalize)
    beds["_status"] = beds["status"].map(normalize)
    beds = beds[beds["_status"].isin(
        {"occupied", "in use", "available", "vacant", "open", "ready", "clean", "empty"}
    )]
    icu_capacity = int((beds["_type"] == "icu").sum())
    emergency_capacity = int((beds["_type"] == "emergency").sum())
    icu_occupied = int(((beds["_type"] == "icu") & (beds["_status"] == "occupied")).sum())
    emergency_occupied = int(
        ((beds["_type"] == "emergency") & (beds["_status"] == "occupied")).sum()
    )
    days = sorted(forecast["date"].astype(str).unique())
    rows = []
    for day in days:
        daily = forecast[forecast["date"].astype(str) == day]
        allocations = bed_allocation_detail[
            bed_allocation_detail["date"].astype(str) == day
        ]
        icu_allocation = allocations[allocations["department"] == "icu"]
        projected_icu = (
            int(icu_allocation["projected_occupied_beds"].iloc[0])
            if not icu_allocation.empty else 0
        )
        icu_demand = projected_icu
        emergency_allocation = allocations[allocations["department"] == "emergency"]
        emergency_demand = (
            int(emergency_allocation["projected_occupied_beds"].iloc[0])
            if not emergency_allocation.empty else 0
        )

        icu_allocated = (
            int(icu_allocation["optimized_allocated"].sum()) if not icu_allocation.empty else 0
        )
        emergency_baseline_allocated = (
            int(emergency_allocation["baseline_allocated"].sum())
            if not emergency_allocation.empty else 0
        )
        emergency_total_allocated = (
            int(emergency_allocation["optimized_allocated"].sum())
            if not emergency_allocation.empty else 0
        )
        emergency_general_allocated = (
            int(emergency_allocation["optimized_emergency_flex_beds"].sum())
            if not emergency_allocation.empty else 0
        )
        emergency_allocated = max(
            0, emergency_total_allocated - emergency_general_allocated
        )
        emergency_staff = staffed_bed_capacity(staff_coverage, "emergency", day)
        icu_staff = staffed_bed_capacity(staff_coverage, "icu", day)
        general_row = daily[daily["department"] == "general medicine"]
        general_capacity = (
            int(general_row["operational_bed_capacity"].max())
            if not general_row.empty else 0
        )
        general_occupancy = (
            float(general_row["projected_occupied_beds"].max())
            if not general_row.empty else 0
        )
        rows.append({
            "date": day,
            "projected_icu_census_beds": icu_demand,
            "icu_capacity": icu_capacity,
            "icu_reserve": min(ICU_RESERVE, icu_capacity),
            "icu_staff_capacity": icu_staff,
            "icu_allocated": icu_allocated,
            "icu_overflow": max(0, icu_demand - icu_allocated),
            "forecast_emergency_bed_demand": emergency_demand,
            "emergency_capacity": emergency_capacity,
            "emergency_reserve": min(EMERGENCY_RESERVE, emergency_capacity),
            "emergency_occupied_today": emergency_occupied,
            "emergency_staff_capacity": emergency_staff,
            "general_medicine_bed_capacity": general_capacity,
            "general_medicine_projected_occupancy": round(general_occupancy, 2),
            "emergency_flexible_beds_used": emergency_general_allocated,
            "emergency_baseline_allocated": emergency_baseline_allocated,
            "emergency_baseline_overflow": max(
                0, emergency_demand - emergency_baseline_allocated
            ),
            "emergency_allocated": emergency_allocated,
            "emergency_general_overflow_allocated": emergency_general_allocated,
            "emergency_total_allocated": emergency_total_allocated,
            "emergency_overflow": max(
                0,
                emergency_demand
                - emergency_total_allocated
            ),
        })
    return pd.DataFrame(rows)


def validate_outputs(
    staff_schedule: pd.DataFrame,
    staff_coverage: pd.DataFrame,
    appointments: pd.DataFrame,
    appointment_metrics: pd.DataFrame,
    bed_detail: pd.DataFrame,
    bed_kpis: pd.DataFrame,
    emergency: pd.DataFrame,
    resource_gaps: pd.DataFrame,
) -> None:
    """Independently check critical resource constraints before publishing outputs."""
    errors = []
    if not staff_coverage.empty and (staff_coverage["unfilled_staff"] > 0).any():
        errors.append("The capacity plan leaves staff coverage positions unfilled.")
    if not staff_schedule.empty:
        if staff_schedule.duplicated(["date", "staff_id"]).any():
            errors.append("A staff member was scheduled more than once on a date.")
        if (staff_schedule["prior_7_day_hours"] + staff_schedule["hours"]
                > staff_schedule["max_hours"]).any():
            errors.append("A scheduled staff member exceeds the weekly hour limit.")
        for _, group in staff_schedule.groupby("staff_id"):
            intervals = [
                _shift_interval(
                    date.fromisoformat(row["date"]),
                    row["shift"],
                    float(row["hours"]),
                )
                for _, row in group.sort_values(["date", "shift"]).iterrows()
            ]
            intervals.sort()
            for previous, current in zip(intervals, intervals[1:]):
                if (current[0] - previous[1]).total_seconds() < 11 * 60 * 60:
                    errors.append("A scheduled staff member violates the minimum rest period.")
                    break
    if (staff_coverage["assigned_staff"] > staff_coverage["required_staff"]).any():
        errors.append("Staff coverage exceeds the requested position count.")
    if not appointment_metrics.empty and int(
        appointment_metrics["optimized_doctor_time_conflicts"].iloc[0]
    ):
        errors.append("The optimized appointment schedule contains overlapping doctor bookings.")
    if not appointment_metrics.empty and int(
        appointment_metrics["optimized_unavailable_doctor_bookings"].iloc[0]
    ):
        errors.append("An appointment was assigned to an unavailable doctor.")
    if not appointment_metrics.empty and int(
        appointment_metrics["optimized_off_shift_doctor_bookings"].iloc[0]
    ):
        errors.append("An appointment was assigned to a doctor outside their shift.")
    if not bed_detail.empty:
        if (bed_detail["optimized_allocated"] > bed_detail["projected_occupied_beds"]).any():
            errors.append("Bed allocations exceed projected department demand.")
        if (bed_detail["optimized_allocated"] > bed_detail["staff_bed_capacity"]).any():
            errors.append("Optimized bed allocations exceed available staffing capacity.")
        if not (bed_detail["optimized_overflow"]
                == bed_detail["projected_occupied_beds"] - bed_detail["optimized_allocated"]).all():
            errors.append("Bed overflow does not reconcile to projected demand.")
        if not (
            bed_detail["optimized_overflow"]
            == bed_detail["optimized_overflow_due_to_staffing"]
            + bed_detail["optimized_overflow_due_to_bed_constraints"]
        ).all():
            errors.append("Optimized bed overflow does not reconcile to its capacity causes.")
    if not resource_gaps.empty:
        if (
            resource_gaps["incremental_nurses_required_per_shift"] < 0
        ).any():
            errors.append("Resource gap analysis reports a negative nurse shortfall.")
        if (
            resource_gaps["optimized_allocated"]
            + resource_gaps["optimized_overflow"]
            != resource_gaps["allocation_demand_beds"]
        ).any():
            errors.append("Resource gap analysis does not reconcile to allocated demand.")
    if not bed_kpis.empty:
        for column in (
            "staffing_constraint_violations",
            "physical_capacity_violations",
            "emergency_icu_reserve_violations",
        ):
            if (bed_kpis[column] > 0).any():
                errors.append(f"Bed allocation has {column.replace('_', ' ')}.")
    if not emergency.empty:
        if (emergency["icu_allocated"] > emergency["icu_capacity"] - emergency["icu_reserve"]).any():
            errors.append("ICU allocation violates the protected reserve.")
        if (emergency["emergency_allocated"] > emergency["emergency_capacity"]).any():
            errors.append("Emergency allocation exceeds physical capacity.")
        if (
            emergency["emergency_allocated"]
            > emergency["emergency_capacity"] - emergency["emergency_reserve"]
        ).any():
            errors.append("Emergency allocation violates the protected reserve.")
        if (emergency["icu_allocated"] > emergency["icu_staff_capacity"]).any():
            errors.append("ICU allocation exceeds staffing capacity.")
        if (
            emergency["emergency_allocated"]
            + emergency["emergency_general_overflow_allocated"]
            > emergency["emergency_staff_capacity"]
        ).any():
            errors.append("Emergency allocation exceeds staffing capacity.")
        if (
            emergency["emergency_general_overflow_allocated"]
            > emergency["emergency_flexible_beds_used"]
        ).any():
            errors.append("Emergency overflow exceeds allocated flexible beds.")
        if (
            emergency["emergency_total_allocated"]
            != emergency["emergency_allocated"]
            + emergency["emergency_general_overflow_allocated"]
        ).any():
            errors.append("Emergency allocation does not reconcile across bed types.")
        if (
            emergency["icu_overflow"]
            != emergency["projected_icu_census_beds"].map(math.ceil)
            - emergency["icu_allocated"]
        ).any():
            errors.append("ICU census overflow does not reconcile to projected occupied beds.")
        if (
            emergency["emergency_overflow"]
            != emergency["forecast_emergency_bed_demand"].map(math.ceil)
            - emergency["emergency_total_allocated"]
        ).any():
            errors.append("Emergency overflow does not reconcile to forecast demand.")
    if errors:
        raise RuntimeError("Resource plan validation failed: " + "; ".join(errors))


def main() -> None:
    subprocess.run(
        [sys.executable, str(RESOURCE_SUPPORT_GENERATOR)],
        check=True,
        cwd=ROOT,
    )
    inputs = load_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    require_fresh = os.environ.get("CAREOPS_REQUIRE_FRESH_DATA", "").strip() == "1"
    input_audit = audit_resource_inputs(inputs, require_fresh=require_fresh)
    input_audit.to_csv(OUTPUT / "data_quality_audit.csv", index=False)
    failed_checks = input_audit[input_audit["status"] == "FAIL"]
    if not failed_checks.empty:
        raise RuntimeError(
            "Resource input quality checks failed: "
            + "; ".join(failed_checks["check"].astype(str))
            + f". Review {OUTPUT / 'data_quality_audit.csv'}."
        )
    backtest = build_forecast_backtest(ROOT)
    if backtest.empty:
        raise RuntimeError("No saved forecast holdout predictions were available to score.")
    backtest.to_csv(OUTPUT / "forecast_backtest_summary.csv", index=False)
    audit_warnings = input_audit.loc[
        input_audit["status"] == "WARN", "check"
    ].astype(str).tolist()
    base_physical_bed_count = len(inputs["beds"])
    base_operational_bed_count = int(
        inputs["beds"]["status"].map(normalize).isin(
            {"occupied", "in use", "available", "vacant", "open", "ready", "clean", "empty"}
        ).sum()
    )
    initial_bed_forecast = build_bed_forecast(inputs)
    surge_beds = build_synthetic_surge_beds(initial_bed_forecast, inputs["beds"])
    if not surge_beds.empty:
        inputs["beds"] = pd.concat([inputs["beds"], surge_beds], ignore_index=True)
    inputs["synthetic_surge_beds"] = surge_beds
    bed_forecast = build_bed_forecast(inputs)
    planning_dates = sorted(
        pd.to_datetime(bed_forecast["date"]).dt.date.unique().tolist()
    )
    planning_date = planning_dates[0]
    staff_schedule, staff_coverage = optimize_staff_schedule(
        inputs, planning_date, planning_dates, bed_forecast
    )
    appointment_schedule, appointment_metrics = optimize_appointments(inputs)
    bed_detail, bed_kpis = optimize_bed_allocations(bed_forecast, inputs, staff_coverage)
    emergency = optimize_emergency_capacity(
        bed_forecast, inputs, staff_coverage, bed_detail
    )
    optimized_bed_kpis = bed_kpis[bed_kpis["scenario"] == "optimized"]
    if int(optimized_bed_kpis["overflow_bed_days"].sum()) != 0:
        raise RuntimeError(
            "Synthetic surge capacity plan did not eliminate optimized bed overflow."
        )
    if int(staff_coverage["unfilled_staff"].sum()) != 0:
        raise RuntimeError(
            "Synthetic surge staffing plan left unfilled shift positions."
        )
    if (
        int(emergency["icu_overflow"].sum()) != 0
        or int(emergency["emergency_overflow"].sum()) != 0
    ):
        raise RuntimeError(
            "Synthetic surge capacity plan did not eliminate ICU/emergency overflow."
        )
    resource_gaps = build_resource_gap_analysis(
        bed_forecast, bed_detail, staff_coverage
    )
    operational_bed_count = int(
        inputs["beds"]["status"].map(normalize).isin(
            {"occupied", "in use", "available", "vacant", "open", "ready", "clean", "empty"}
        ).sum()
    )
    horizon_days = len(planning_dates)
    capacity_plan_rows: list[dict[str, Any]] = []
    if not surge_beds.empty:
        for department, added_beds in surge_beds.groupby("department").size().items():
            capacity_plan_rows.append({
                "resource_type": "temporary_surge_beds",
                "department": department,
                "role": "",
                "shift": "",
                "added_quantity": int(added_beds),
                "forecast_horizon_days": horizon_days,
                "scheduled_hours": 0,
                "unit_cost_usd": 250.0,
                "planned_cost_usd": float(added_beds * horizon_days * 250.0),
                "cost_basis": "illustrative bed-day rate; scenario assumption",
                "scenario_only": True,
            })
    surge_pool = inputs.get("synthetic_surge_staff_plan", pd.DataFrame())
    scheduled_surge = (
        staff_schedule[staff_schedule["staff_source"] == "synthetic_surge_pool"]
        if not staff_schedule.empty else pd.DataFrame()
    )
    if not surge_pool.empty:
        if scheduled_surge.empty:
            scheduled_surge = pd.DataFrame(columns=[
                "department", "role", "shift", "staff_id", "hours", "hourly_rate_usd",
            ])
        scheduled_summary = scheduled_surge.groupby(
            ["department", "role", "shift"], as_index=False
        ).agg(
            scheduled_surge_shifts=("staff_id", "size"),
            scheduled_hours=("hours", "sum"),
            scheduled_staff=("staff_id", "nunique"),
        )
        staff_plan = surge_pool.merge(
            scheduled_summary,
            on=["department", "role", "shift"],
            how="left",
        ).fillna({
            "scheduled_surge_shifts": 0,
            "scheduled_hours": 0,
            "scheduled_staff": 0,
        })
        for _, row in staff_plan.iterrows():
            hours = float(row["scheduled_hours"])
            rate = float(row["hourly_rate_usd"])
            capacity_plan_rows.append({
                "resource_type": "synthetic_surge_staff",
                "department": row["department"],
                "role": row["role"],
                "shift": row["shift"],
                "added_quantity": int(row["synthetic_surge_pool_staff"]),
                "forecast_horizon_days": horizon_days,
                "scheduled_hours": hours,
                "unit_cost_usd": rate,
                "planned_cost_usd": hours * rate,
                "cost_basis": "illustrative hourly rate applied to scheduled surge hours",
                "scenario_only": True,
                "peak_required_staff_per_shift": int(row["peak_required_staff"]),
                "scheduled_surge_staff": int(row["scheduled_staff"]),
            })
    capacity_plan = pd.DataFrame(capacity_plan_rows)
    mean_daily_nurse_gap = float(
        resource_gaps.groupby("date")["incremental_nurses_required_per_shift"].sum().mean()
    )
    no_show_backtest = backtest[backtest["model"] == "no_show_xgboost"]
    if no_show_backtest.empty:
        no_show_validation_summary = "No saved no-show holdout predictions were available."
    else:
        no_show_score = no_show_backtest.iloc[0]
        if pd.notna(no_show_score["pre_holdout_baseline_brier"]):
            no_show_validation_summary = (
                f"No-show Brier score: {no_show_score['brier_score']:.4f} versus "
                f"{no_show_score['pre_holdout_baseline_brier']:.4f} pre-holdout "
                "prevalence baseline; "
                + (
                    "model probabilities outperform the baseline."
                    if bool(no_show_score["brier_beats_baseline"])
                    else "model probabilities do not beat the baseline."
                )
            )
        else:
            no_show_validation_summary = (
                f"No-show Brier score: {no_show_score['brier_score']:.4f}; "
                "no leakage-safe historical baseline was available."
            )
    validate_outputs(
        staff_schedule, staff_coverage, appointment_schedule, appointment_metrics,
        bed_detail, bed_kpis, emergency, resource_gaps,
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = {
        "bed_availability_forecast.csv": bed_forecast,
        "bed_allocation_simulation.csv": bed_detail,
        "bed_allocation_kpis.csv": bed_kpis,
        "optimized_staff_schedule.csv": staff_schedule,
        "staff_coverage_kpis.csv": staff_coverage,
        "optimized_appointment_schedule.csv": appointment_schedule,
        "appointment_optimization_kpis.csv": appointment_metrics,
        "emergency_capacity_optimization.csv": emergency,
        "resource_gap_analysis.csv": resource_gaps,
        "capacity_expansion_plan.csv": capacity_plan,
    }
    for filename, frame in outputs.items():
        frame.to_csv(OUTPUT / filename, index=False)
    report = [
        "CAREOPS AI - RESOURCE OPTIMIZATION SCENARIO",
        "Synthetic-data planning aid; human review is required before operational use.",
        f"Staff planning horizon: {planning_date.isoformat()} to "
        f"{planning_dates[-1].isoformat()}",
        f"Run date: {datetime.now().date().isoformat()}",
        "Data scope: bundled synthetic/historical snapshot; not a live hospital feed.",
        f"Input audit warnings: {int((input_audit['status'] == 'WARN').sum())}; "
        f"failed checks: {int((input_audit['status'] == 'FAIL').sum())}.",
        "Input warnings: " + ", ".join(audit_warnings),
        f"Forecast holdout checks scored: {len(backtest)} model/group combinations.",
        f"Regression holdout comparisons where the model beat its baseline: "
        f"{int((backtest['beats_baseline'] == True).sum())} of "
        f"{int(backtest['beats_baseline'].notna().sum())}.",
        "No-show probability calibration is evaluated separately with Brier score.",
        "Holdout scores evaluate saved historical test predictions; they do not "
        "validate future operational performance.",
        no_show_validation_summary,
        "No-show risk scores are retained for review but do not influence slot "
        "movement until a leakage-safe calibration evaluation shows improvement "
        "over a pre-holdout baseline.",
        f"Appointment planning dates: "
        f"{appointment_metrics['planning_date_start'].iloc[0]} to "
        f"{appointment_metrics['planning_date_end'].iloc[0]}",
        "",
        "Bed allocation KPIs:",
        bed_kpis.to_string(index=False),
        "",
        "Appointment KPIs:",
        appointment_metrics.to_string(index=False),
        "",
        f"Staff positions unfilled: {int(staff_coverage['unfilled_staff'].sum())}",
        f"Forecast mean occupancy: "
        f"{bed_kpis['mean_forecast_occupancy_utilization'].iloc[0]:.1%} of "
        f"{operational_bed_count} planned operational beds.",
        f"Mean incremental nurses needed per shift across departments: "
        f"{mean_daily_nurse_gap:.1f} at the "
        f"{NURSE_TO_BED_CAPACITY}:1 scenario ratio.",
        f"Optimized overflow attribution: "
        f"{int(bed_kpis.loc[bed_kpis['scenario'] == 'optimized', 'overflow_bed_days_due_to_staffing'].iloc[0])} "
        "bed-days due to staffing limits and "
        f"{int(bed_kpis.loc[bed_kpis['scenario'] == 'optimized', 'overflow_bed_days_due_to_bed_constraints'].iloc[0])} "
        "bed-days due to dedicated bed/compatibility limits.",
        f"Emergency capacity forecast days: {len(emergency)}",
        f"Emergency baseline overflow bed-days: {int(emergency['emergency_baseline_overflow'].sum())}",
        f"Emergency optimized overflow bed-days: {int(emergency['emergency_overflow'].sum())}",
        f"Emergency overflow served in General Medicine beds: "
        f"{int(emergency['emergency_general_overflow_allocated'].sum())} bed-days",
        "Capacity assumptions: one assigned nurse supports up to "
        f"{NURSE_TO_BED_CAPACITY} beds for scenario analysis; "
        f"{ICU_RESERVE} ICU and {EMERGENCY_RESERVE} emergency beds are protected.",
        "Department occupancy is calibrated to the hospital-wide occupancy forecast; "
        "uncalibrated admission/LOS projections are retained in the bed forecast output.",
        "Forecast occupancy utilization is based on projected census divided by "
        "physical operational beds. Allocation utilization and staffed-capacity "
        "utilization are reported separately; neither is a measured patient wait time.",
        "The occupancy forecast was trained against the original inventory "
        f"({base_physical_bed_count} physical / {base_operational_bed_count} operational beds). "
        f"The scenario adds {len(surge_beds)} temporary synthetic surge beds, for "
        f"{len(inputs['beds'])} planned physical beds and {operational_bed_count} planned "
        "operational beds. Post-expansion allocation utilization uses planned capacity.",
        "The synthetic flex workforce and temporary beds are explicit scenario-only "
        "capacity assumptions, not confirmed hires, funded beds, or live resources.",
        "Illustrative resource-expansion costs are scenario assumptions, not market quotes.",
        "Optimized capacity gate: zero unfilled staffing positions, zero bed overflow, "
        "zero ICU overflow, and zero emergency overflow.",
        "Emergency staffing capacity uses the optimized daily shift coverage "
        "for each forecast date.",
    ]
    (OUTPUT / "resource_optimization_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Resource optimization outputs written to {OUTPUT}")
    print(bed_kpis.to_string(index=False))
    print(f"Staffing coverage: {int(staff_coverage['assigned_staff'].sum())} assigned, "
          f"{int(staff_coverage['unfilled_staff'].sum())} unfilled shift positions.")
    print(f"Temporary synthetic surge beds added: {len(surge_beds)}; "
          f"synthetic surge workforce pool: "
          f"{int(surge_pool['synthetic_surge_pool_staff'].sum()) if not surge_pool.empty else 0}.")
    print(f"Optimized overflow gate: bed={int(optimized_bed_kpis['overflow_bed_days'].sum())}, "
          f"ICU={int(emergency['icu_overflow'].sum())}, "
          f"emergency={int(emergency['emergency_overflow'].sum())}.")
    print(f"Appointment scheduling: {int((appointment_schedule['status'] == 'SCHEDULED').sum())}"
          f"/{len(appointment_schedule)} scheduled; doctor conflicts "
          f"{int(appointment_metrics['baseline_doctor_time_conflicts'].iloc[0])} -> "
          f"{int(appointment_metrics['optimized_doctor_time_conflicts'].iloc[0])}.")
    print(f"Emergency/ICU forecast days optimized: {len(emergency)}.")


if __name__ == "__main__":
    main()
