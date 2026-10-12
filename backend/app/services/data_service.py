from __future__ import annotations

import base64
import csv
import json
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOTS = ("data", "ml", "optimization")
SAMPLE_SIZE = 500
DATE_COLUMN_MARKERS = ("date", "datetime", "timestamp", "_at")
REQUIRED_COLUMNS = {
    "data/synthetic/patients.csv": {"patient_id"},
    "data/synthetic/admissions.csv": {"admission_id", "patient_id", "admission_date"},
    "data/synthetic/appointments.csv": {"appointment_id", "date"},
    "data/synthetic/beds.csv": {"bed_id", "department", "status"},
    "data/synthetic/staff.csv": {"staff_id", "department", "role"},
    "data/synthetic/shifts.csv": {"shift_id", "date", "staff_id"},
    "data/synthetic/daily_operations.csv": {"date", "department"},
}
OPERATIONAL_STATUSES = {
    "occupied": {"occupied", "in use"},
    "available": {"available", "vacant", "open", "ready", "clean", "empty"},
    "unavailable": {"maintenance", "closed", "out of service"},
}


def _dataset_id(relative_path: str) -> str:
    return base64.urlsafe_b64encode(relative_path.encode("utf-8")).decode("ascii").rstrip("=")


def _decode_dataset_id(dataset_id: str) -> str:
    try:
        padding = "=" * (-len(dataset_id) % 4)
        relative_path = base64.b64decode(
            dataset_id + padding,
            altchars=b"-_",
            validate=True,
        ).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("Unknown dataset.") from exc

    candidate = PurePosixPath(relative_path)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.suffix.casefold() != ".csv"
        or not candidate.parts
        or candidate.parts[0] not in DATASET_ROOTS
    ):
        raise ValueError("Unknown dataset.")
    return candidate.as_posix()


def _date_columns(columns: list[str]) -> list[str]:
    return [
        column for column in columns
        if any(marker in column.casefold() for marker in DATE_COLUMN_MARKERS)
        or column.casefold() == "date"
    ]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
    except ValueError:
        for date_format in ("%m/%d/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(normalized[:10], date_format).date()
            except ValueError:
                continue
    return None


def _business_meaning(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    name = path.stem.replace("_", " ")
    if relative_path.startswith("data/synthetic/resource_support/"):
        return f"Synthetic support inputs for resource planning: {name}."
    if relative_path.startswith("data/synthetic/"):
        return f"Generated synthetic hospital operations data: {name}."
    if relative_path.startswith("data/processed/"):
        return f"Processed hospital operations dataset or model output: {name}."
    if relative_path.startswith("data/feature_engineering/"):
        return f"Feature-engineering dataset: {name}."
    if relative_path.startswith("data/features/"):
        return f"Model or optimization feature dataset: {name}."
    if relative_path.startswith("data/eda/"):
        return f"Exploratory data analysis output: {name}."
    if relative_path.startswith("ml/"):
        return f"Machine-learning input, prediction, evaluation, or output artifact: {name}."
    if relative_path.startswith("optimization/"):
        return f"Optimization input, result, or evaluation artifact: {name}."
    return f"Project dataset: {name}."


def _quality_sample(path: Path, columns: list[str], date_columns: list[str]) -> dict[str, int]:
    sample = pd.read_csv(path, nrows=SAMPLE_SIZE, low_memory=False)
    missing_values = int(sample.isna().sum().sum())
    duplicate_rows = int(sample.duplicated().sum()) if not sample.empty else 0
    invalid_date_values = 0
    for column in date_columns:
        if column in sample:
            values = sample[column].dropna().astype(str)
            invalid_date_values += sum(_parse_date(value) is None for value in values)
    return {
        "sample_rows": len(sample),
        "missing_values_in_sample": missing_values,
        "duplicate_rows_in_sample": duplicate_rows,
        "invalid_date_values_in_sample": invalid_date_values,
    }


@lru_cache(maxsize=16)
def _read_frame(absolute_path: str, size_bytes: int, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(absolute_path, low_memory=False)


@lru_cache(maxsize=512)
def _catalog_entry(
    absolute_path: str,
    relative_path: str,
    size_bytes: int,
    modified_ns: int,
) -> dict[str, Any]:
    path = Path(absolute_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        columns = next(reader, [])
        row_count = 0
        date_columns = _date_columns(columns)
        date_column_indexes = {
            column: index for index, column in enumerate(columns)
            if column in date_columns
        }
        date_min: dict[str, date] = {}
        date_max: dict[str, date] = {}
        for values in reader:
            row_count += 1
            for column in date_columns:
                column_index = date_column_indexes[column]
                if column_index >= len(values):
                    continue
                value = values[column_index]
                parsed = _parse_date(value)
                if parsed is not None:
                    date_min[column] = min(date_min.get(column, parsed), parsed)
                    date_max[column] = max(date_max.get(column, parsed), parsed)

    sample = pd.read_csv(path, nrows=SAMPLE_SIZE, low_memory=False)
    data_types = {
        column: ("date" if column in date_columns else str(sample[column].dtype))
        for column in columns
        if column in sample.columns
    }
    sample_quality = _quality_sample(path, columns, date_columns)
    relative_posix = PurePosixPath(relative_path).as_posix()
    return {
        "id": _dataset_id(relative_posix),
        "source_file": relative_posix,
        "display_name": Path(relative_path).stem.replace("_", " ").title(),
        "business_meaning": _business_meaning(relative_posix),
        "rows": row_count,
        "columns": columns,
        "column_count": len(columns),
        "data_types": data_types,
        "date_ranges": {
            column: {
                "min": date_min[column].isoformat() if column in date_min else None,
                "max": date_max[column].isoformat() if column in date_max else None,
            }
            for column in date_columns
        },
        "size_bytes": size_bytes,
        "modified_ns": modified_ns,
        "modified_at": datetime.fromtimestamp(
            modified_ns / 1_000_000_000,
            tz=timezone.utc,
        ).isoformat(),
        "sample_quality": sample_quality,
    }


def _iter_csv_files(project_root: Path):
    for root_name in DATASET_ROOTS:
        root = project_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.csv"):
            if any(part in {"__pycache__", ".pytest_cache", "node_modules"} for part in path.parts):
                continue
            if path.is_file():
                yield path


def get_dataset_catalog(
    search: str | None = None,
    area: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> list[dict[str, Any]]:
    normalized_search = search.strip().casefold() if search else ""
    normalized_area = area.strip().casefold() if area else ""
    entries = []
    for path in _iter_csv_files(project_root):
        relative_path = path.relative_to(project_root).as_posix()
        stat = path.stat()
        entry = _catalog_entry(
            str(path.resolve()),
            relative_path,
            stat.st_size,
            stat.st_mtime_ns,
        )
        searchable_text = " ".join([
            relative_path.replace("_", " "),
            entry["business_meaning"],
            " ".join(entry["columns"]).replace("_", " "),
        ]).casefold()
        if normalized_search and normalized_search not in searchable_text:
            continue
        if normalized_area and PurePosixPath(relative_path).parts[0].casefold() != normalized_area:
            continue
        entries.append(entry)
    return sorted(entries, key=lambda item: item["source_file"])


def get_dataset_page(
    dataset_id: str,
    offset: int = 0,
    limit: int = 100,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    relative_path = _decode_dataset_id(dataset_id)
    path = (project_root / Path(*PurePosixPath(relative_path).parts)).resolve()
    if not path.is_relative_to(project_root.resolve()) or not path.is_file():
        raise ValueError("Unknown dataset.")

    stat = path.stat()
    entry = _catalog_entry(
        str(path),
        relative_path,
        stat.st_size,
        stat.st_mtime_ns,
    )
    if offset >= entry["rows"]:
        rows: list[dict[str, Any]] = []
    else:
        frame = pd.read_csv(
            path,
            skiprows=range(1, offset + 1),
            nrows=limit,
            low_memory=False,
        )
        rows = json.loads(frame.to_json(orient="records", date_format="iso"))
    return {
        "dataset": entry,
        "offset": offset,
        "limit": limit,
        "returned_rows": len(rows),
        "rows": rows,
    }


def get_dataset_validation(project_root: Path = PROJECT_ROOT) -> list[dict[str, Any]]:
    results = []
    for dataset in get_dataset_catalog(project_root=project_root):
        required = REQUIRED_COLUMNS.get(dataset["source_file"], set())
        missing_required = sorted(required - set(dataset["columns"]))
        sample = dataset["sample_quality"]
        failures = []
        warnings = []
        if missing_required:
            failures.append("Missing required columns: " + ", ".join(missing_required))
        if dataset["rows"] == 0:
            warnings.append("Dataset has a header but no data rows.")
        if sample["missing_values_in_sample"]:
            warnings.append(
                f"{sample['missing_values_in_sample']} missing values in the first "
                f"{sample['sample_rows']} rows."
            )
        if sample["duplicate_rows_in_sample"]:
            warnings.append(
                f"{sample['duplicate_rows_in_sample']} duplicate rows in the first "
                f"{sample['sample_rows']} rows."
            )
        if sample["invalid_date_values_in_sample"]:
            warnings.append(
                f"{sample['invalid_date_values_in_sample']} unparseable date values "
                f"in the first {sample['sample_rows']} rows."
            )
        results.append({
            "dataset_id": dataset["id"],
            "source_file": dataset["source_file"],
            "status": "fail" if failures else "warn" if warnings else "pass",
            "required_columns_checked": sorted(required),
            "missing_required_columns": missing_required,
            "sample_rows_checked": sample["sample_rows"],
            "failures": failures,
            "warnings": warnings,
        })
    return results


def get_departments(project_root: Path = PROJECT_ROOT) -> list[str]:
    departments: set[str] = set()
    for relative_path in (
        "data/synthetic/admissions.csv",
        "data/synthetic/appointments.csv",
        "data/synthetic/beds.csv",
        "data/synthetic/staff.csv",
    ):
        path = project_root / Path(*PurePosixPath(relative_path).parts)
        if not path.is_file():
            continue
        stat = path.stat()
        frame = _read_frame(str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if "department" in frame:
            departments.update(
                frame["department"].dropna().astype(str).str.strip().loc[lambda values: values.ne("")]
            )
    return sorted(departments, key=str.casefold)


def _filter_dates(
    frame: pd.DataFrame,
    column: str,
    start_date: date | None,
    end_date: date | None,
    department: str | None,
) -> pd.DataFrame:
    filtered = frame.copy()
    if column in filtered:
        filtered[column] = pd.to_datetime(filtered[column], errors="coerce")
        filtered = filtered.dropna(subset=[column])
        if start_date is not None:
            filtered = filtered[filtered[column].dt.date >= start_date]
        if end_date is not None:
            filtered = filtered[filtered[column].dt.date <= end_date]
    if department and "department" in filtered:
        filtered = filtered[
            filtered["department"].astype(str).str.casefold() == department.casefold()
        ]
    return filtered


def _load_project_frame(project_root: Path, relative_path: str) -> pd.DataFrame:
    path = project_root / Path(*PurePosixPath(relative_path).parts)
    if not path.is_file():
        return pd.DataFrame()
    stat = path.stat()
    return _read_frame(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def get_operations_analytics(
    start_date: date | None = None,
    end_date: date | None = None,
    department: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    operations = _filter_dates(
        _load_project_frame(project_root, "data/synthetic/daily_operations.csv"),
        "date",
        start_date,
        end_date,
        department,
    )
    appointments = _filter_dates(
        _load_project_frame(project_root, "data/synthetic/appointments.csv"),
        "date",
        start_date,
        end_date,
        department,
    )
    admissions = _filter_dates(
        _load_project_frame(project_root, "data/synthetic/admissions.csv"),
        "admission_date",
        start_date,
        end_date,
        department,
    )

    beds = _load_project_frame(project_root, "data/synthetic/beds.csv")
    staff = _load_project_frame(project_root, "data/synthetic/staff.csv")
    if department:
        if not beds.empty and "department" in beds:
            beds = beds[beds["department"].astype(str).str.casefold() == department.casefold()]
        if not staff.empty and "department" in staff:
            staff = staff[staff["department"].astype(str).str.casefold() == department.casefold()]

    daily_fields = [
        column for column in (
            "admissions", "discharges", "emergency_cases", "ICU_admissions",
            "staff_demand", "bed_demand",
        )
        if column in operations
    ]
    # ``occupancy`` is a per-department percentage, so it is never summed;
    # ``bed_demand`` is the occupied-bed count and sums to the hospital census.
    if operations.empty:
        daily = pd.DataFrame(columns=["date", *daily_fields])
        department_summary = pd.DataFrame(columns=["department"])
    else:
        daily = (
            operations.assign(date=operations["date"].dt.strftime("%Y-%m-%d"))
            .groupby("date", as_index=False)[daily_fields]
            .sum()
            .sort_values("date")
        )
        if "bed_demand" in daily:
            daily["occupied_beds"] = daily["bed_demand"]
        department_summary = (
            operations.groupby("department", as_index=False)
            .agg(
                admissions=("admissions", "sum"),
                discharges=("discharges", "sum"),
                mean_occupancy_percent=("occupancy", "mean"),
                emergency_cases=("emergency_cases", "sum"),
            )
        )

    if not appointments.empty:
        appointments["date"] = appointments["date"].dt.strftime("%Y-%m-%d")
        appointment_trend = (
            appointments.groupby("date", as_index=False)
            .agg(
                appointments=("appointment_id", "size"),
                no_shows=("no_show", "sum"),
            )
        )
        appointment_trend["appointments"] = appointment_trend["appointments"].astype(int)
        appointment_trend["no_shows"] = appointment_trend["no_shows"].astype(int)
        appointment_trend["date"] = appointment_trend["date"].astype(str)
        daily = daily.merge(appointment_trend, on="date", how="outer")
    else:
        daily["appointments"] = 0
        daily["no_shows"] = 0
    daily = daily.sort_values("date").fillna(0)
    for column in ("admissions", "discharges", "emergency_cases", "ICU_admissions", "bed_demand", "appointments", "no_shows"):
        if column in daily:
            daily[column] = daily[column].round().astype(int)

    if not department_summary.empty:
        appointment_departments = (
            appointments.groupby("department", as_index=False)
            .agg(
                appointments=("appointment_id", "size"),
                no_shows=("no_show", "sum"),
            )
        )
        department_summary = department_summary.merge(
            appointment_departments,
            on="department",
            how="outer",
        )

    bed_rows: list[dict[str, Any]] = []
    if not beds.empty and {"department", "status"}.issubset(beds.columns):
        beds = beds.copy()
        beds["_status"] = beds["status"].astype(str).str.strip().str.casefold()
        for name, group in beds.groupby("department", dropna=False):
            statuses = group["_status"]
            total = int(len(group))
            occupied = int(statuses.isin(OPERATIONAL_STATUSES["occupied"]).sum())
            available = int(statuses.isin(OPERATIONAL_STATUSES["available"]).sum())
            bed_rows.append({
                "department": str(name),
                "total_beds": total,
                "occupied_beds": occupied,
                "available_beds": available,
                "occupancy_rate": round(occupied / total * 100, 2) if total else None,
            })
    bed_summary = pd.DataFrame(bed_rows)

    if department_summary.empty:
        department_summary = bed_summary
    elif not bed_summary.empty:
        department_summary = department_summary.merge(
            bed_summary,
            on="department",
            how="outer",
        )

    if not staff.empty and "department" in staff:
        staff_summary = (
            staff.groupby("department", as_index=False)
            .size()
            .rename(columns={"size": "staff_count"})
        )
        if department_summary.empty:
            department_summary = staff_summary
        else:
            department_summary = department_summary.merge(
                staff_summary,
                on="department",
                how="outer",
            )

    if not department_summary.empty:
        for column in department_summary.select_dtypes(include="number").columns:
            department_summary[column] = department_summary[column].fillna(0)
        department_summary = department_summary.sort_values("department", key=lambda values: values.astype(str).str.casefold())

    occupied_beds = 0
    available_beds = 0
    if not beds.empty and "status" in beds:
        bed_status = beds["status"].astype(str).str.strip().str.casefold()
        occupied_beds = int(bed_status.isin(OPERATIONAL_STATUSES["occupied"]).sum())
        available_beds = int(bed_status.isin(OPERATIONAL_STATUSES["available"]).sum())
    no_show_count = int(pd.to_numeric(appointments.get("no_show", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    appointment_count = int(len(appointments))
    date_values = operations["date"] if "date" in operations else pd.Series(dtype="datetime64[ns]")

    return {
        "filters": {
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "department": department,
        },
        "summary": {
            "patients_in_source_dataset": int(
                len(_load_project_frame(project_root, "data/synthetic/patients.csv"))
            ),
            "admissions": int(len(admissions)),
            "appointments": appointment_count,
            "no_shows": no_show_count,
            "beds": int(len(beds)),
            "occupied_beds": occupied_beds,
            "available_beds": available_beds,
            "occupancy_rate": round(occupied_beds / len(beds) * 100, 2) if len(beds) else None,
            "staff": int(len(staff)),
            "date_start": date_values.min().date().isoformat() if not date_values.empty else None,
            "date_end": date_values.max().date().isoformat() if not date_values.empty else None,
        },
        "daily": daily.to_dict(orient="records"),
        "departments": department_summary.to_dict(orient="records"),
    }
