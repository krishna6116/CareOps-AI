"""Write docs/DATA_DICTIONARY.md from the actual source and output files.

Column types, null rates, example values and row counts are read from the
CSVs; the descriptions below are maintained by hand. Run after the pipeline:

    python data/build_data_dictionary.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "DATA_DICTIONARY.md"

DATASETS = [
    ("data/synthetic/patients.csv", "One row per synthetic patient.", "patient_id", {
        "patient_id": "Primary key.", "age": "Years, 0-100.",
        "department": "Department of the patient's most recent care.",
        "severity": "Low / Moderate / High / Critical.",
    }),
    ("data/synthetic/admissions.csv", "One row per inpatient stay. `stay_status` is `in_progress` for patients still in hospital on the cutoff date.", "admission_id", {
        "patient_id": "FK -> patients.patient_id.",
        "assigned_bed_id": "FK -> beds.bed_id. No two stays overlap on one bed.",
        "LOS": "discharge_date - admission_date in days (checked).",
        "ICU_required": "1 if the stay needs an ICU bed; such stays are placed in ICU beds.",
        "discharge_date": "Planned or actual; after the cutoff for in-progress stays.",
    }),
    ("data/synthetic/beds.csv", "Bed inventory snapshot on the cutoff date (230 beds).", "bed_id", {
        "status": "Occupied / Available / Maintenance on the snapshot date.",
        "bed_type": "General / ICU / Emergency / Isolation.",
        "isolation_capable": "Single room usable for isolation.",
        "emergency_reserve": "ICU and emergency beds subject to the protected reserve.",
        "available_for_emergency": "General Medicine beds usable for emergency overflow.",
    }),
    ("data/synthetic/staff.csv", "Staff roster (300).", "staff_id", {
        "availability": "Snapshot availability; not a schedule.",
        "max_hours": "Weekly hour limit used by the scheduler.",
    }),
    ("data/synthetic/shifts.csv", "Historical rostered shifts. Not tied to ward census (see README limitations).", "shift_id", {
        "staff_id": "FK -> staff.staff_id.", "assigned": "Yes / No.",
        "rest_violation": "1 if the shift breaks the minimum rest rule.",
    }),
    ("data/synthetic/appointments.csv", "Outpatient appointments.", "appointment_id", {
        "doctor_id": "FK -> staff.staff_id.",
        "no_show": "Label: 1 if the patient did not attend. Known only after the appointment; never used as a feature.",
        "duration": "Minutes.",
    }),
    ("data/synthetic/daily_operations.csv", "Daily totals per department.", None, {
        "occupancy": "Occupied beds at end of day.",
        "staff_demand": "Synthetic staffing requirement.",
    }),
    ("data/synthetic/resource_support/admission_isolation_labels.csv", "Synthetic isolation flag per admission (scenario probability, not clinical).", "admission_id", {
        "admission_id": "FK -> admissions.admission_id (one to one).",
    }),
    ("ml/admission_forecasting/hospital_30_day_admission_forecast.csv", "Published 30-day hospital admission forecast.", None, {
        "predicted_admissions": "Whole-number expected admissions.",
        "lower_80": "Split-conformal interval bounds (also upper_80, lower_95, upper_95).",
        "model": "Published model; a naive baseline if no learned model won the backtest.",
    }),
    ("ml/occupancy_forecasting/occupancy_future_forecast.csv", "Published 30-day occupied-bed forecast.", None, {
        "forecast_occupied_beds": "Whole beds, capped at physical capacity.",
        "alert_level": "Rule-based capacity alert level.",
    }),
    ("ml/icu_demand_forecasting/icu_future_forecast.csv", "Published ICU admissions forecast (admissions per day, not census).", None, {}),
    ("ml/los_prediction/los_predictions.csv", "Holdout LOS predictions made at admission time.", "admission_id", {
        "predicted_los_days": "Expected LOS in days (continuous).",
    }),
    ("ml/no_show_prediction/no_show_predictions.csv", "Holdout no-show probabilities, scored the day before the appointment.", "appointment_id", {
        "predicted_no_show_probability": "Calibrated probability; a planning signal, not a determination.",
    }),
    ("optimization/resource_planning/outputs/bed_availability_forecast.csv", "Department/day bed projection used by the planner.", None, {
        "projected_occupied_beds": "Current census + forecast admissions - expected discharges, whole beds.",
        "forecast_available_beds": "operational_bed_capacity - projected_occupied_beds, never below 0.",
        "operational_bed_capacity": "In-service beds, including scenario surge beds.",
    }),
    ("optimization/resource_planning/outputs/bed_assignment_requests.csv", "Request-level replay: one row per arrival in the window, with both policies' results.", "admission_id", {
        "compatible_free_beds": "In-service compatible beds free at the start of the day.",
        "optimized_reason": "Reason code when unassigned (see bed_assignment_summary.json).",
        "optimized_placement": "home or off_service.",
    }),
    ("optimization/resource_planning/outputs/bed_assignment_kpis.csv", "Replay KPIs per policy.", "policy", {}),
    ("optimization/resource_planning/outputs/staff_coverage_kpis.csv", "Required vs assigned staff per date, department, shift and role.", None, {}),
]

RELATIONSHIPS = [
    ("admissions.patient_id", "patients.patient_id", "data/synthetic/admissions.csv", "patient_id", "data/synthetic/patients.csv", "patient_id"),
    ("admissions.assigned_bed_id", "beds.bed_id", "data/synthetic/admissions.csv", "assigned_bed_id", "data/synthetic/beds.csv", "bed_id"),
    ("shifts.staff_id", "staff.staff_id", "data/synthetic/shifts.csv", "staff_id", "data/synthetic/staff.csv", "staff_id"),
    ("appointments.doctor_id", "staff.staff_id", "data/synthetic/appointments.csv", "doctor_id", "data/synthetic/staff.csv", "staff_id"),
    ("isolation_labels.admission_id", "admissions.admission_id", "data/synthetic/resource_support/admission_isolation_labels.csv", "admission_id", "data/synthetic/admissions.csv", "admission_id"),
]


def example(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return ""
    value = values.iloc[0]
    text = f"{value:.4g}" if isinstance(value, float) else str(value)
    return text.replace("|", "/")[:40]


def describe(relative: str, summary: str, key: str | None, notes: dict[str, str]) -> list[str]:
    frame = pd.read_csv(ROOT / relative, low_memory=False)
    lines = [f"### `{relative}`", "", summary, ""]
    facts = [f"{len(frame):,} rows, {frame.shape[1]} columns"]
    if key:
        unique = frame[key].is_unique
        facts.append(f"key `{key}` {'unique' if unique else 'NOT unique'}")
    for column in ("date", "admission_date", "planning_date", "snapshot_date"):
        if column in frame:
            dates = pd.to_datetime(frame[column], errors="coerce").dropna()
            if not dates.empty:
                facts.append(f"`{column}` {dates.min().date()} to {dates.max().date()}")
            break
    lines += ["; ".join(facts) + ".", "", "| Column | Type | Null % | Example | Notes |", "|---|---|---:|---|---|"]
    for column in frame.columns:
        nulls = frame[column].isna().mean() * 100
        lines.append(
            f"| `{column}` | {frame[column].dtype} | {nulls:.1f} | {example(frame[column])} | {notes.get(column, '')} |"
        )
    return lines + [""]


def relationship_lines() -> list[str]:
    lines = ["## Relationships (checked on generation)", "", "| Foreign key | References | Orphan rows |", "|---|---|---:|"]
    for child, parent, child_path, child_col, parent_path, parent_col in RELATIONSHIPS:
        child_values = pd.read_csv(ROOT / child_path, usecols=[child_col])[child_col]
        parent_values = set(pd.read_csv(ROOT / parent_path, usecols=[parent_col])[parent_col])
        orphans = int((~child_values.isin(parent_values)).sum())
        lines.append(f"| `{child}` | `{parent}` | {orphans} |")
    return lines + [""]


def main() -> None:
    lines = [
        "# Data dictionary",
        "",
        "Generated by `data/build_data_dictionary.py` from the committed files; do not edit by hand.",
        "All records are synthetic. Types are as pandas reads the CSV.",
        "",
    ]
    lines += relationship_lines()
    lines += ["## Datasets", ""]
    for relative, summary, key, notes in DATASETS:
        lines += describe(relative, summary, key, notes)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
