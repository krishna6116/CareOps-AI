"""Data-quality and retrospective forecast checks for the resource planner."""

from __future__ import annotations

import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BACKTEST_FILES = {
    "admissions": Path("ml/admission_forecasting/admission_test_predictions.csv"),
    "occupancy": Path("ml/occupancy_forecasting/occupancy_test_predictions.csv"),
    "icu": Path("ml/icu_demand_forecasting/icu_demand_test_predictions.csv"),
    "los": Path("ml/los_prediction/los_predictions.csv"),
    "no_show": Path("ml/no_show_prediction/no_show_predictions.csv"),
}


def _result(
    checks: list[dict[str, Any]],
    check: str,
    status: str,
    value: Any,
    details: str,
) -> None:
    checks.append({
        "check": check,
        "status": status,
        "value": value,
        "details": details,
    })


def _date_bounds(frame: pd.DataFrame, column: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    parsed = pd.to_datetime(frame[column], errors="coerce").dropna()
    if parsed.empty:
        return pd.NaT, pd.NaT
    return parsed.min().normalize(), parsed.max().normalize()


def audit_resource_inputs(
    inputs: dict[str, pd.DataFrame],
    run_date: date | None = None,
    require_fresh: bool = False,
) -> pd.DataFrame:
    """Create a human-readable audit; structural errors are marked FAIL."""
    today = run_date or datetime.now().date()
    checks: list[dict[str, Any]] = []
    for name, frame in inputs.items():
        if name == "no_show_predictions" and frame.empty:
            _result(
                checks, f"{name}_not_empty", "WARN", 0,
                "Optional no-show predictions are absent; appointment risk defaults apply.",
            )
            continue
        if frame.empty:
            _result(checks, f"{name}_not_empty", "FAIL", 0, "Required input has no rows.")
        else:
            _result(checks, f"{name}_not_empty", "PASS", len(frame), "Input contains rows.")

    no_show = inputs.get("no_show_predictions", pd.DataFrame())
    if not no_show.empty:
        required_no_show_columns = {
            "appointment_id", "predicted_no_show_probability",
        }
        missing_columns = required_no_show_columns - set(no_show.columns)
        if missing_columns:
            _result(
                checks, "no_show_prediction_schema", "FAIL",
                ", ".join(sorted(missing_columns)),
                "No-show prediction rows require appointment IDs and probabilities.",
            )
        else:
            duplicate_predictions = int(
                no_show["appointment_id"].duplicated(keep=False).sum()
            )
            probabilities = pd.to_numeric(
                no_show["predicted_no_show_probability"], errors="coerce"
            )
            invalid_probabilities = int(
                (probabilities.isna() | ~np.isfinite(probabilities)
                 | (probabilities < 0) | (probabilities > 1)).sum()
            )
            _result(
                checks, "no_show_prediction_duplicate_ids",
                "FAIL" if duplicate_predictions else "PASS", duplicate_predictions,
                "Each appointment may have at most one no-show score."
                if duplicate_predictions else "No-show prediction IDs are unique.",
            )
            _result(
                checks, "no_show_prediction_probability_bounds",
                "FAIL" if invalid_probabilities else "PASS", invalid_probabilities,
                "No-show probabilities must be finite and within [0, 1]."
                if invalid_probabilities else "No-show probabilities are finite and bounded.",
            )
    _result(
        checks, "data_scope", "WARN", "synthetic/historical",
        "Bundled data is synthetic and historical; results do not establish live-site performance.",
    )

    identifier_columns = {
        "beds": "bed_id",
        "staff": "staff_id",
        "shifts": "shift_id",
        "appointments": "appointment_id",
        "admissions": "admission_id",
        "isolation_labels": "admission_id",
    }
    for name, column in identifier_columns.items():
        if name not in inputs:
            continue
        frame = inputs[name]
        duplicates = int(frame[column].duplicated(keep=False).sum())
        _result(
            checks, f"{name}_duplicate_{column}",
            "FAIL" if duplicates else "PASS", duplicates,
            "Duplicate identifiers invalidate unique resource assignments."
            if duplicates else "Identifiers are unique.",
        )

    for name, frame, columns in (
        ("daily_operations", inputs["daily_operations"], ["date", "department"]),
        ("admission_forecast", inputs["admission_forecast"], ["date", "department"]),
        ("occupancy_forecast", inputs["occupancy_forecast"], ["date"]),
        ("icu_forecast", inputs["icu_forecast"], ["date"]),
        ("isolation_forecast", inputs.get("isolation_forecast", pd.DataFrame()), ["date", "department"]),
    ):
        normalized = frame.copy()
        if "department" in columns:
            normalized["department"] = normalized["department"].astype(str).str.casefold().str.strip()
            normalized = normalized[normalized["department"] != "all departments"]
        duplicates = int(normalized.duplicated(columns, keep=False).sum())
        _result(
            checks, f"{name}_duplicate_grain",
            "FAIL" if duplicates else "PASS", duplicates,
            f"Duplicate rows at expected grain {columns}."
            if duplicates else f"Rows are unique at expected grain {columns}.",
        )

    operations = inputs["daily_operations"]
    if not operations.empty:
        start, end = _date_bounds(operations, "date")
        if pd.isna(end):
            _result(checks, "operations_date_validity", "FAIL", "no valid dates", "No usable operational dates.")
        else:
            lag = (today - end.date()).days
            stale = lag > 2
            _result(
                checks, "latest_observed_operations_age_days",
                "FAIL" if stale and require_fresh else "WARN" if stale else "PASS", lag,
                f"Latest operational date is {end.date()}; expected a current operational feed "
                "for live planning." if lag > 2 else f"Operational history ends {end.date()}.",
            )
            expected_days = (end - start).days + 1
            observed_days = operations.assign(
                _date=pd.to_datetime(operations["date"], errors="coerce").dt.normalize()
            )["_date"].nunique()
            missing_days = max(0, expected_days - observed_days)
            _result(
                checks, "operations_calendar_gaps",
                "WARN" if missing_days else "PASS", missing_days,
                "At least one calendar date is absent from operational history."
                if missing_days else "Operational history has no missing calendar dates.",
            )
            operations_by_day = operations.assign(
                _date=pd.to_datetime(operations["date"], errors="coerce").dt.normalize()
            ).groupby("_date")["department"].nunique()
            inconsistent_days = int(
                (operations_by_day != operations["department"].nunique()).sum()
            )
            _result(
                checks, "operations_department_day_coverage",
                "WARN" if inconsistent_days else "PASS", inconsistent_days,
                "Some operational dates are missing one or more departments."
                if inconsistent_days else "Each operational date has all listed departments.",
            )

    forecasts = inputs["admission_forecast"]
    if not forecasts.empty:
        start, end = _date_bounds(forecasts, "date")
        forecast_age = (today - start.date()).days if not pd.isna(start) else None
        _result(
            checks, "forecast_start_age_days",
            "FAIL" if require_fresh and (forecast_age is None or forecast_age > 0)
            else "WARN" if forecast_age is None or forecast_age > 0 else "PASS",
            forecast_age,
            f"Forecast starts {start.date()}; {max(0, forecast_age or 0)} forecast day(s) "
            "have elapsed at run time. Regenerate before operational use."
            if forecast_age is not None and forecast_age > 0
            else f"Forecast window is {start.date()} through {end.date()}.",
        )
        detailed_forecasts = forecasts[
            forecasts["department"].astype(str).str.casefold().str.strip()
            != "all departments"
        ]
        daily_counts = detailed_forecasts.assign(
            _date=pd.to_datetime(detailed_forecasts["date"], errors="coerce").dt.normalize()
        ).groupby("_date")["department"].nunique()
        _result(
            checks, "admission_forecast_department_coverage",
            "PASS" if not daily_counts.empty and daily_counts.nunique() == 1 else "WARN",
            int(daily_counts.min()) if not daily_counts.empty else 0,
            "Detailed department count is consistent across forecast dates."
            if not daily_counts.empty and daily_counts.nunique() == 1
            else "Department coverage varies by forecast date.",
        )
        expected_start = pd.to_datetime(operations["date"], errors="coerce").max().normalize() + pd.Timedelta(days=1)
        origin_gap = (start - expected_start).days if not pd.isna(start) else None
        _result(
            checks, "forecast_origin_alignment",
            "PASS" if origin_gap == 0 else "WARN",
            origin_gap,
            "Forecast begins the day after latest operational history."
            if origin_gap == 0
            else f"Forecast starts {origin_gap} day(s) from the expected next day after "
                 f"operational history ({expected_start.date()}).",
        )
        admission_dates = pd.to_datetime(forecasts["date"], errors="coerce").dropna().dt.normalize().unique()
        admission_dates = pd.DatetimeIndex(admission_dates).sort_values()
        expected_forecast_days = (
            (admission_dates.max() - admission_dates.min()).days + 1
            if len(admission_dates) else 0
        )
        admission_date_gaps = max(0, expected_forecast_days - len(admission_dates))
        _result(
            checks, "admission_forecast_calendar_gaps",
            "WARN" if admission_date_gaps else "PASS", admission_date_gaps,
            "Admission forecast has missing calendar dates."
            if admission_date_gaps else "Admission forecast dates are continuous.",
        )
        expected_date_set = set(pd.Timestamp(value) for value in admission_dates)
        icu_dates = pd.to_datetime(
            inputs["icu_forecast"]["date"], errors="coerce"
        ).dropna().dt.normalize().unique()
        icu_date_set = set(pd.Timestamp(value) for value in icu_dates)
        missing_icu_days = len(expected_date_set - icu_date_set)
        extra_icu_days = len(icu_date_set - expected_date_set)
        _result(
            checks, "icu_forecast_horizon_alignment",
            "FAIL" if missing_icu_days or extra_icu_days else "PASS",
            f"{missing_icu_days} missing / {extra_icu_days} extra",
            "ICU admissions forecast dates must exactly match the admission forecast horizon."
            if missing_icu_days or extra_icu_days
            else "ICU admissions forecast covers the exact admission forecast horizon.",
        )
        icu_units = set(
            inputs["icu_forecast"].get(
                "forecast_target_unit", pd.Series(dtype=str)
            ).astype(str).str.casefold()
        )
        valid_icu_units = icu_units == {"admissions_per_day"}
        _result(
            checks, "icu_forecast_target_semantics",
            "PASS" if valid_icu_units else "FAIL",
            ", ".join(sorted(icu_units)),
            "ICU model target is explicitly daily admissions, not occupied beds."
            if valid_icu_units
            else "ICU forecast target unit is missing or inconsistent.",
        )
        occupancy_dates = pd.to_datetime(
            inputs["occupancy_forecast"]["date"], errors="coerce"
        ).dropna().dt.normalize().unique()
        occupancy_date_set = set(pd.Timestamp(value) for value in occupancy_dates)
        missing_occupancy_days = len(expected_date_set - occupancy_date_set)
        extra_occupancy_days = len(occupancy_date_set - expected_date_set)
        _result(
            checks, "occupancy_forecast_horizon_alignment",
            "FAIL" if missing_occupancy_days or extra_occupancy_days else "PASS",
            f"{missing_occupancy_days} missing / {extra_occupancy_days} extra",
            "Occupancy forecast dates must exactly match the admission forecast horizon."
            if missing_occupancy_days or extra_occupancy_days
            else "Occupancy forecast covers the exact admission forecast horizon.",
        )

    occupancy = inputs["occupancy_forecast"]
    if not occupancy.empty and "forecast_occupancy_rate" in occupancy.columns:
        physical_beds = len(inputs["beds"])
        operational_statuses = {
            "occupied", "in use", "available", "vacant", "open", "ready", "clean", "empty",
        }
        operational_beds = int(
            inputs["beds"]["status"].astype(str).str.casefold().str.strip()
            .isin(operational_statuses).sum()
        )
        occupied = pd.to_numeric(occupancy["forecast_occupied_beds"], errors="coerce")
        rate = pd.to_numeric(occupancy["forecast_occupancy_rate"], errors="coerce")
        discrepancy = float((occupied - rate * physical_beds).abs().max())
        _result(
            checks, "occupancy_forecast_capacity_basis",
            "WARN" if physical_beds != operational_beds else "PASS",
            f"{operational_beds} operational / {physical_beds} physical; "
            f"max rate-to-count difference {discrepancy:.2f} beds",
            "Occupancy forecast uses the full physical inventory, while allocation "
            "excludes non-operational beds; interpret the two denominators separately."
            if physical_beds != operational_beds
            else "Forecast occupancy and optimizer use the same bed inventory basis.",
        )

    for name, frame in (("bed_inventory", inputs["beds"]), ("staff_availability", inputs["staff"])):
        snapshot_date_column = "as_of_date" if "as_of_date" in frame.columns else "date"
        if snapshot_date_column not in frame.columns:
            _result(
                checks, f"{name}_snapshot_date",
                "FAIL" if require_fresh else "WARN", "missing",
                f"{name.replace('_', ' ').title()} has no snapshot timestamp.",
            )
            continue
        snapshot_dates = pd.to_datetime(frame[snapshot_date_column], errors="coerce")
        if snapshot_dates.isna().any():
            _result(
                checks, f"{name}_snapshot_date",
                "FAIL", int(snapshot_dates.isna().sum()),
                "Some snapshot records have invalid or missing dates.",
            )
            continue
        unique_dates = snapshot_dates.dt.normalize().nunique()
        latest_snapshot = snapshot_dates.max().date()
        age = (today - latest_snapshot).days
        provenance = (
            set(frame["data_provenance"].astype(str).str.casefold())
            if "data_provenance" in frame.columns else {"unknown"}
        )
        synthetic_source = any(
            "synthetic" in value or "assumed" in value for value in provenance
        )
        strict_invalid = require_fresh and (
            age > 2 or unique_dates != 1 or synthetic_source
        )
        status = (
            "FAIL" if strict_invalid
            else "WARN" if age > 2 or unique_dates != 1 or synthetic_source
            else "PASS"
        )
        _result(
            checks, f"{name}_snapshot_date", status, latest_snapshot.isoformat(),
            f"Snapshot age {age} day(s); provenance: {', '.join(sorted(provenance))}."
            + (
                " Synthetic/assumed snapshots are not live data."
                if synthetic_source else ""
            ),
        )

    valid_availability = {"available", "on call", "busy", "leave", "unavailable"}
    unknown_availability = sorted(
        set(inputs["staff"]["availability"].astype(str).str.casefold().str.strip())
        - valid_availability
    )
    _result(
        checks, "staff_availability_values",
        "WARN" if unknown_availability else "PASS",
        ", ".join(unknown_availability) if unknown_availability else 0,
        "Unknown availability labels may be excluded from scheduling."
        if unknown_availability else "Availability labels are recognized.",
    )

    appointments = inputs["appointments"]
    if not appointments.empty:
        _, latest_appointment = _date_bounds(appointments, "date")
        lag = (today - latest_appointment.date()).days if not pd.isna(latest_appointment) else None
        _result(
            checks, "latest_appointment_data_age_days",
            "FAIL" if require_fresh and (lag is None or lag > 1)
            else "WARN" if lag is None or lag > 1 else "PASS", lag,
            f"Latest appointment date is {latest_appointment.date()}; this is historical "
            "schedule data, not a live forward appointment book."
            if lag is not None and lag > 1 else "Appointment input is current.",
        )

    for name, frame, columns in (
        ("admission_forecast_numeric_values", forecasts, ["predicted_admissions"]),
        ("isolation_forecast_numeric_values", inputs["isolation_forecast"],
         ["predicted_isolation_admissions", "historical_smoothed_isolation_rate"]),
        ("occupancy_forecast_numeric_values", inputs["occupancy_forecast"], ["forecast_occupied_beds"]),
        ("icu_forecast_numeric_values", inputs["icu_forecast"], ["predicted_icu_admissions"]),
    ):
        values = frame[columns].apply(pd.to_numeric, errors="coerce")
        invalid = int((~np.isfinite(values.to_numpy(dtype=float))).sum())
        negative = int((values < 0).sum().sum())
        failures = invalid + negative
        _result(
            checks, name, "FAIL" if failures else "PASS", failures,
            "Forecast values must be finite and non-negative."
            if failures else "Forecast values are finite and non-negative.",
        )

    admissions = inputs.get("admissions", pd.DataFrame())
    isolation_labels = inputs.get("isolation_labels", pd.DataFrame())
    if not isolation_labels.empty:
        label_values = isolation_labels["isolation_required"].astype(str).str.casefold()
        invalid_labels = int((~label_values.isin({"true", "false", "1", "0"})).sum())
        source_values = set(isolation_labels["label_source"].astype(str))
        synthetic_labels = any("synthetic" in value.casefold() for value in source_values)
        _result(
            checks, "isolation_label_integrity",
            "FAIL" if invalid_labels else "PASS", invalid_labels,
            "Isolation labels are not binary." if invalid_labels
            else "Isolation labels are binary and carry source metadata.",
        )
        _result(
            checks, "isolation_label_provenance",
            "WARN" if synthetic_labels else "PASS",
            ", ".join(sorted(source_values)),
            "Isolation labels are synthetic scenario values, not clinical determinations."
            if synthetic_labels else "Isolation labels have non-synthetic provenance.",
        )
    isolation_forecast = inputs.get("isolation_forecast", pd.DataFrame())
    if not isolation_forecast.empty:
        rates = pd.to_numeric(
            isolation_forecast["historical_smoothed_isolation_rate"], errors="coerce"
        )
        invalid_rates = int((rates.isna() | (rates < 0) | (rates > 1)).sum())
        excess = 0
        if "predicted_admissions" in forecasts.columns:
            joined = isolation_forecast.merge(
                forecasts[["date", "department", "predicted_admissions"]],
                on=["date", "department"],
                how="left",
                validate="one_to_one",
            )
            excess = int((
                pd.to_numeric(joined["predicted_isolation_admissions"], errors="coerce")
                > pd.to_numeric(joined["predicted_admissions"], errors="coerce") + 1e-9
            ).sum())
        failures = invalid_rates + excess
        _result(
            checks, "isolation_forecast_bounds",
            "FAIL" if failures else "PASS", failures,
            "Isolation probabilities must be in [0, 1] and isolation arrivals "
            "cannot exceed total arrivals."
            if failures else "Isolation scenario rates/arrivals are bounded by total arrivals.",
        )
    if not admissions.empty and {"admission_date", "discharge_date"} <= set(admissions.columns):
        admitted = pd.to_datetime(admissions["admission_date"], errors="coerce")
        discharged = pd.to_datetime(admissions["discharge_date"], errors="coerce")
        invalid_dates = int((admitted.isna() | discharged.isna()).sum())
        impossible_stays = int((discharged < admitted).sum())
        failures = invalid_dates + impossible_stays
        _result(
            checks, "admission_stay_date_integrity",
            "FAIL" if failures else "PASS", failures,
            "Admission/discharge dates are missing or discharge precedes admission."
            if failures else "Admission and discharge dates are valid.",
        )
        if "isolation_required" not in admissions.columns:
            _result(
                checks, "isolation_demand_source", "WARN", "missing",
                "Admission records contain no explicit isolation requirement; "
                "isolation need cannot be inferred safely.",
            )

    reserve_beds = inputs["beds"]
    if "emergency_reserve" in reserve_beds.columns:
        flagged = int(
            reserve_beds["emergency_reserve"].astype(str).str.casefold().isin(
                {"1", "true", "yes", "y"}
            ).sum()
        )
        _result(
            checks, "emergency_reserve_flag_semantics",
            "WARN" if flagged else "PASS", flagged,
            "Bed inventory labels reserve-capable beds, but the meaning and protected "
            "count are not documented; scenario reserve values are configured separately."
            if flagged else "No beds are explicitly flagged as emergency reserve.",
        )

    return pd.DataFrame(checks)


def _score(
    model: str,
    actual: pd.Series,
    predicted: pd.Series,
    baseline: pd.Series | None,
    start: Any,
    end: Any,
    group: str,
) -> dict[str, Any]:
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    prediction_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(prediction_values)
    actual_values = actual_values[valid]
    prediction_values = prediction_values[valid]
    if not len(actual_values):
        raise ValueError(f"No finite retrospective predictions are available for {model}.")
    errors = prediction_values - actual_values
    denominator = float(np.abs(actual_values).sum())
    metrics: dict[str, Any] = {
        "model": model,
        "evaluation_type": "saved chronological holdout; not a live prospective trial",
        "group": group,
        "test_rows": int(len(actual_values)),
        "test_start": str(pd.Timestamp(start).date()) if pd.notna(start) else "",
        "test_end": str(pd.Timestamp(end).date()) if pd.notna(end) else "",
        "mae": round(float(np.abs(errors).mean()), 6),
        "rmse": round(float(np.sqrt(np.square(errors).mean())), 6),
        "wape_percent": round(100 * float(np.abs(errors).sum()) / denominator, 4) if denominator else np.nan,
        "baseline_mae": np.nan,
        "baseline_rmse": np.nan,
        "mae_change_vs_baseline_percent": np.nan,
        "beats_baseline": pd.NA,
        "brier_score": np.nan,
        "pre_holdout_baseline_brier": np.nan,
        "brier_change_vs_baseline_percent": np.nan,
        "brier_beats_baseline": pd.NA,
    }
    if baseline is not None:
        baseline_values = pd.to_numeric(baseline, errors="coerce").to_numpy(dtype=float)[valid]
        baseline_valid = np.isfinite(baseline_values)
        if baseline_valid.all():
            baseline_errors = baseline_values - actual_values
            baseline_mae = float(np.abs(baseline_errors).mean())
            metrics["baseline_mae"] = round(baseline_mae, 6)
            metrics["baseline_rmse"] = round(
                float(np.sqrt(np.square(baseline_errors).mean())), 6
            )
            metrics["mae_change_vs_baseline_percent"] = (
                round(100 * (float(np.abs(errors).mean()) - baseline_mae) / baseline_mae, 4)
                if baseline_mae else np.nan
            )
            metrics["beats_baseline"] = bool(float(np.abs(errors).mean()) < baseline_mae)
    return metrics


def _binary_roc_auc(actual: np.ndarray, probability: np.ndarray) -> float:
    positive = actual == 1
    positive_count = int(positive.sum())
    negative_count = len(actual) - positive_count
    if not positive_count or not negative_count:
        return np.nan
    ranks = pd.Series(probability).rank(method="average").to_numpy(dtype=float)
    positive_rank_sum = float(ranks[positive].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def build_forecast_backtest(
    root: Path,
    files: dict[str, Path] | None = None,
) -> pd.DataFrame:
    """Score saved chronological holdout predictions, including simple baselines."""
    paths = {**DEFAULT_BACKTEST_FILES, **(files or {})}
    rows: list[dict[str, Any]] = []

    def read(name: str) -> pd.DataFrame | None:
        path = paths[name]
        if not path.is_absolute():
            path = root / path
        return pd.read_csv(path) if path.exists() else None

    admissions = read("admissions")
    if admissions is not None:
        required = {"date", "department", "actual_admissions", "ml_prediction", "baseline_prediction"}
        missing = required - set(admissions.columns)
        if missing:
            raise ValueError(f"Admission holdout file missing columns: {sorted(missing)}")
        admissions["date"] = pd.to_datetime(admissions["date"], errors="coerce")
        for department, group in admissions.groupby("department"):
            rows.append(_score(
                "admissions_xgboost", group["actual_admissions"],
                group["ml_prediction"], group["baseline_prediction"],
                group["date"].min(), group["date"].max(), str(department),
            ))
        daily = admissions.groupby("date", as_index=False)[
            ["actual_admissions", "ml_prediction", "baseline_prediction"]
        ].sum()
        rows.append(_score(
            "admissions_xgboost", daily["actual_admissions"],
            daily["ml_prediction"], daily["baseline_prediction"],
            daily["date"].min(), daily["date"].max(), "hospital_total",
        ))

    occupancy = read("occupancy")
    if occupancy is not None:
        date_column = "date"
        occupancy[date_column] = pd.to_datetime(occupancy[date_column], errors="coerce")
        if {"actual_occupancy_rate", "predicted_occupancy_rate", "seasonal_naive_rate"} <= set(occupancy.columns):
            rows.append(_score(
                "occupancy_ridge", occupancy["actual_occupancy_rate"],
                occupancy["predicted_occupancy_rate"], occupancy["seasonal_naive_rate"],
                occupancy[date_column].min(), occupancy[date_column].max(), "hospital",
            ))
        if {"actual_occupied_beds", "predicted_occupied_beds"} <= set(occupancy.columns):
            rows.append(_score(
                "occupancy_ridge_bed_count", occupancy["actual_occupied_beds"],
                occupancy["predicted_occupied_beds"], None,
                occupancy[date_column].min(), occupancy[date_column].max(), "hospital",
            ))

    icu = read("icu")
    if icu is not None:
        icu["date"] = pd.to_datetime(icu["date"], errors="coerce")
        rows.append(_score(
            "icu_xgboost", icu["actual_icu_demand"],
            icu["predicted_icu_demand"], icu.get("baseline_7_day"),
            icu["date"].min(), icu["date"].max(), "icu",
        ))

    los = read("los")
    if los is not None:
        date_column = "admission_datetime"
        los[date_column] = pd.to_datetime(los[date_column], errors="coerce")
        for department, group in los.groupby("department"):
            rows.append(_score(
                "los_xgboost", group["actual_los_days"],
                group["predicted_los_days"], None,
                group[date_column].min(), group[date_column].max(), str(department),
            ))
        rows.append(_score(
            "los_xgboost", los["actual_los_days"],
            los["predicted_los_days"], None,
            los[date_column].min(), los[date_column].max(), "all_departments",
        ))

    no_show = read("no_show")
    if no_show is not None:
        actual = pd.to_numeric(no_show["actual_no_show"], errors="coerce").to_numpy(dtype=float)
        probability = pd.to_numeric(
            no_show["predicted_no_show_probability"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(probability)
        actual, probability = actual[valid], probability[valid]
        if len(actual) and (
            not np.isin(actual, [0, 1]).all()
            or (probability < 0).any()
            or (probability > 1).any()
        ):
            raise ValueError(
                "No-show holdout labels must be binary and probabilities within [0, 1]."
            )
        if len(actual):
            holdout_dates = pd.to_datetime(
                no_show.loc[valid, "appointment_datetime"], errors="coerce"
            )
            holdout_start = holdout_dates.min()
            data_dir = Path(
                os.environ.get("CAREOPS_DATA_DIR", root / "data" / "processed")
            )
            history_path = data_dir / "appointments.csv"
            prior_prevalence = np.nan
            if history_path.exists():
                history = pd.read_csv(history_path)
                if {"date", "no_show"} <= set(history.columns):
                    history_dates = pd.to_datetime(history["date"], errors="coerce")
                    history_labels = pd.to_numeric(history["no_show"], errors="coerce")
                    prior = history[
                        (history_dates < holdout_start)
                        & history_labels.isin([0, 1])
                    ]
                    if not prior.empty:
                        prior_prevalence = float(
                            pd.to_numeric(prior["no_show"], errors="coerce").mean()
                        )
            brier = float(np.square(probability - actual).mean())
            roc_auc = _binary_roc_auc(actual, probability)
            baseline_brier = (
                float(np.square(prior_prevalence - actual).mean())
                if np.isfinite(prior_prevalence) else np.nan
            )
            rows.append({
                "model": "no_show_xgboost",
                "evaluation_type": "saved chronological holdout; not a live prospective trial",
                "group": "holdout",
                "test_rows": int(len(actual)),
                "test_start": str(holdout_start.date()),
                "test_end": str(holdout_dates.max().date()),
                "mae": np.nan,
                "rmse": np.nan,
                "wape_percent": np.nan,
                "baseline_mae": np.nan,
                "baseline_rmse": np.nan,
                "mae_change_vs_baseline_percent": np.nan,
                "beats_baseline": pd.NA,
                "brier_score": round(brier, 6),
                "roc_auc": round(roc_auc, 6) if np.isfinite(roc_auc) else np.nan,
                "pre_holdout_baseline_brier": (
                    round(baseline_brier, 6) if np.isfinite(baseline_brier) else np.nan
                ),
                "brier_change_vs_baseline_percent": (
                    round(100 * (brier - baseline_brier) / baseline_brier, 4)
                    if np.isfinite(baseline_brier) and baseline_brier else np.nan
                ),
                "brier_beats_baseline": (
                    bool(brier < baseline_brier)
                    if np.isfinite(baseline_brier) else pd.NA
                ),
            })
    return pd.DataFrame(rows)
