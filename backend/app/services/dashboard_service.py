from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from backend.app.database import SessionLocal
from backend.app.models.careops import AlertLog

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_DIR = PROJECT_ROOT / "optimization" / "resource_planning" / "outputs"
ML_DIR = PROJECT_ROOT / "ml"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_overview() -> dict[str, object]:
    beds = _read_csv(DATA_DIR / "beds.csv")
    total = int(len(beds)) if not beds.empty else 0
    statuses = beds["status"].astype(str).str.strip().str.casefold() if not beds.empty else pd.Series(dtype=str)
    occupied = int(statuses.isin({"occupied", "in use"}).sum())
    available = int(statuses.isin({"available", "vacant", "open", "ready", "clean", "empty"}).sum())
    unavailable = int(statuses.isin({"maintenance", "closed", "out of service"}).sum())
    utilization = round((occupied / total) * 100, 2) if total else 0.0

    forecast = _read_csv(ML_DIR / "admission_forecasting" / "admission_30_day_forecast.csv")
    occupancy = _read_csv(ML_DIR / "occupancy_forecasting" / "occupancy_future_forecast.csv")
    icu_beds = beds[beds["bed_type"].astype(str).str.casefold().eq("icu")] if not beds.empty else pd.DataFrame()
    emergency_beds = beds[beds["bed_type"].astype(str).str.casefold().eq("emergency")] if not beds.empty else pd.DataFrame()
    # The forecast file carries department rows plus an "All Departments"
    # total row per day; count department rows only.
    department_forecast = (
        forecast[forecast["department"].astype(str).str.casefold() != "all departments"]
        if not forecast.empty else forecast
    )
    admissions = (
        int(pd.to_numeric(department_forecast["predicted_admissions"], errors="coerce").fillna(0).sum())
        if not department_forecast.empty else 0
    )
    predicted_occupancy = (
        int(pd.to_numeric(occupancy["forecast_occupied_beds"], errors="coerce").iloc[-1])
        if not occupancy.empty else None
    )
    forecast_dates = (
        pd.to_datetime(forecast["date"], errors="coerce").dropna()
        if not forecast.empty else pd.Series(dtype="datetime64[ns]")
    )

    manifest = _read_manifest()
    alerts = get_alerts()
    recommendations = get_recommendations()

    return {
        "total_beds": total,
        "occupied_beds": occupied,
        "available_beds": available,
        "unavailable_beds": unavailable,
        "utilization": utilization,
        "predicted_admissions": admissions,
        "predicted_occupancy": predicted_occupancy,
        "icu_total_beds": int(len(icu_beds)),
        "icu_occupied_beds": int(icu_beds["status"].astype(str).str.casefold().isin({"occupied", "in use"}).sum()) if not icu_beds.empty else 0,
        "icu_available_beds": int(icu_beds["status"].astype(str).str.casefold().isin({"available", "vacant", "open", "ready", "clean", "empty"}).sum()) if not icu_beds.empty else 0,
        "emergency_total_beds": int(len(emergency_beds)),
        "emergency_available_beds": int(emergency_beds["status"].astype(str).str.casefold().isin({"available", "vacant", "open", "ready", "clean", "empty"}).sum()) if not emergency_beds.empty else 0,
        "alerts": len(alerts),
        "optimization_status": manifest.get("pipeline_status", "unknown"),
        "operational_recommendations": recommendations[:3],
        "scenario": manifest.get("scenario_as_of_date", "unknown"),
        "latest_forecast_date": forecast_dates.max().date().isoformat() if not forecast_dates.empty else None,
        "forecast_horizon_days": int(manifest.get("forecast_horizon_days", 0)),
        "latest_optimization_run": manifest.get("scenario_as_of_date"),
    }


def _read_manifest() -> dict[str, object]:
    manifest_path = OUTPUT_DIR / "portfolio_run_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def get_bed_management() -> list[dict[str, object]]:
    beds = _read_csv(DATA_DIR / "beds.csv")
    if beds.empty:
        return []
    beds["_status"] = beds["status"].astype(str).str.strip().str.casefold()
    rows = []
    latest_forecast = _read_csv(OUTPUT_DIR / "bed_availability_forecast.csv")
    if not latest_forecast.empty:
        latest_forecast["_date"] = pd.to_datetime(latest_forecast["date"], errors="coerce")
        latest_date = latest_forecast["_date"].max()
        latest_forecast = latest_forecast[latest_forecast["_date"] == latest_date]
    forecast_by_department = (
        latest_forecast.set_index("department").to_dict(orient="index")
        if not latest_forecast.empty else {}
    )
    for department, group in beds.groupby("department", dropna=False):
        statuses = group["_status"]
        forecast = forecast_by_department.get(str(department).casefold(), {})
        total = len(group)
        occupied = int(statuses.isin({"occupied", "in use"}).sum())
        available = int(statuses.isin({"available", "vacant", "open", "ready", "clean", "empty"}).sum())
        reserved = int(statuses.eq("reserved").sum())
        unavailable = int(statuses.isin({"maintenance", "closed", "out of service"}).sum())
        rows.append({
            "department": department,
            "total_beds": total,
            "occupied_beds": occupied,
            "available_beds": available,
            "reserved_beds": reserved,
            "unavailable_beds": unavailable,
            "utilization": round(occupied / total * 100, 2) if total else 0.0,
            "forecast_date": forecast.get("date"),
            "projected_occupied_beds": forecast.get("projected_occupied_beds"),
            "incoming_admissions": forecast.get("predicted_admissions"),
            "expected_discharges": forecast.get("expected_discharges"),
            "forecast_available_beds": forecast.get("forecast_available_beds"),
            "operational_capacity": forecast.get("operational_bed_capacity"),
        })
    return sorted(rows, key=lambda row: str(row["department"]))


def get_alerts() -> list[dict[str, object]]:
    audit_path = OUTPUT_DIR / "data_quality_audit.csv"
    if audit_path.exists():
        audit = _read_csv(audit_path)
        if not audit.empty:
            rows = audit[audit["status"].isin(["FAIL", "WARN"])].copy()
            rows["severity"] = rows["status"].map({"FAIL": "high", "WARN": "medium"})
            manifest = _read_manifest()
            rows["timestamp"] = manifest.get("generated_at", manifest.get("scenario_as_of_date", "unknown"))
            rows["source_run"] = "audit"
            rows["message"] = rows.get("details", rows.get("message", rows["check"]))
            rows["category"] = rows["check"]
            rows["status"] = rows["status"].str.lower()
            rows["source"] = "data_quality_audit"
            audit_alerts = rows[["severity", "category", "message", "timestamp", "source_run", "status", "source"]].to_dict(orient="records")
        else:
            audit_alerts = []
    else:
        audit_alerts = []
    with SessionLocal() as session:
        stored_alerts = session.query(AlertLog).order_by(AlertLog.created_at.desc()).limit(500).all()
    persisted = [{
        "severity": alert.severity,
        "category": alert.category,
        "message": alert.message,
        "timestamp": alert.created_at.isoformat() if alert.created_at else "unknown",
        "source_run": alert.run_id,
        "status": "open",
        "source": alert.source or "manual",
        "id": alert.id,
    } for alert in stored_alerts]
    if audit_alerts or persisted:
        return persisted + audit_alerts
    manifest = _read_manifest()
    if manifest.get("next_process_blockers"):
        return [{
            "severity": "high",
            "category": "pipeline",
            "message": "; ".join(manifest["next_process_blockers"]),
            "timestamp": manifest.get("generated_at", manifest.get("scenario_as_of_date", "unknown")),
            "source_run": manifest.get("scenario_as_of_date", "unknown"),
            "status": "open",
            "source": "portfolio_manifest",
        }]
    return []


def get_recommendations() -> list[dict[str, object]]:
    gap_path = OUTPUT_DIR / "resource_gap_analysis.csv"
    if gap_path.exists():
        gap = _read_csv(gap_path)
        if not gap.empty:
            gaps = gap.copy()
            gaps["date"] = pd.to_datetime(gaps["date"], errors="coerce")
            gaps = gaps.sort_values(
                ["incremental_nurses_required_per_shift", "optimized_overflow", "date"],
                ascending=[False, False, True],
            )
            gaps["date"] = gaps["date"].dt.date.astype(str)
            results = gaps.head(10).to_dict(orient="records")
            for recommendation in results:
                overflow = int(recommendation.get("optimized_overflow", 0) or 0)
                nurses_needed = int(
                    recommendation.get("incremental_nurses_required_per_shift", 0) or 0
                )
                if overflow > 0:
                    recommendation.update({
                        "title": "Address projected bed overflow",
                        "priority": "high",
                        "detail": (
                            f"{overflow} projected overflow bed(s) on "
                            f"{recommendation.get('date', 'an unknown date')} in "
                            f"{recommendation.get('department', 'an unknown department')}."
                        ),
                    })
                elif nurses_needed > 0:
                    recommendation.update({
                        "title": "Review forecast nurse coverage",
                        "priority": "medium",
                        "detail": (
                            f"{nurses_needed} additional nurse(s) per shift are indicated "
                            f"for {recommendation.get('department', 'an unknown department')} "
                            f"on {recommendation.get('date', 'an unknown date')}."
                        ),
                    })
                else:
                    recommendation.update({
                        "title": "Monitor forecast resource demand",
                        "priority": "low",
                        "detail": (
                            f"No bed overflow or incremental nurse gap was reported for "
                            f"{recommendation.get('department', 'an unknown department')} "
                            f"on {recommendation.get('date', 'an unknown date')}."
                        ),
                    })
            return results
    return []


def create_alert(payload: dict[str, str | None]) -> dict[str, object]:
    manifest = _read_manifest()
    with SessionLocal() as session:
        alert = AlertLog(
            run_id=str(manifest.get("scenario_as_of_date", "unscoped")),
            severity=str(payload["severity"]),
            category=str(payload["category"]),
            message=str(payload["message"]),
            source=payload.get("source") or "manual",
        )
        session.add(alert)
        session.commit()
        session.refresh(alert)
        return {
            "id": alert.id,
            "severity": alert.severity,
            "category": alert.category,
            "message": alert.message,
            "source": alert.source,
            "status": "open",
            "source_run": alert.run_id,
            "timestamp": alert.created_at.isoformat() if alert.created_at else None,
        }


def get_run_history() -> list[dict[str, object]]:
    manifest = _read_manifest()
    if not manifest:
        return []
    return [{
        "run_id": str(manifest.get("scenario_as_of_date", "unknown")),
        "date_time": str(manifest.get("scenario_as_of_date", "unknown")),
        "scenario": str(manifest.get("scenario_as_of_date", "unknown")),
        "optimizer_status": manifest.get("pipeline_status", "unknown"),
        "kpis": manifest.get("optimized_capacity_gates", {}),
        "recommendations": get_recommendations(),
    }]


def get_staff_dashboard() -> dict[str, Any]:
    staff = _read_csv(DATA_DIR / "staff.csv")
    schedule = _read_csv(OUTPUT_DIR / "optimized_staff_schedule.csv")
    coverage = _read_csv(OUTPUT_DIR / "staff_coverage_kpis.csv")
    return {
        "staff_count": int(len(staff)),
        "scheduled_shift_count": int(len(schedule)),
        "scheduled_hours": float(pd.to_numeric(schedule.get("hours", pd.Series(dtype=float)), errors="coerce").sum()),
        "unfilled_positions": int(pd.to_numeric(coverage.get("unfilled_staff", pd.Series(dtype=float)), errors="coerce").sum()),
        "staff": staff.to_dict(orient="records"),
        "schedule": schedule.to_dict(orient="records"),
        "coverage": coverage.to_dict(orient="records"),
        "policy_comparison": _read_csv(OUTPUT_DIR / "staff_schedule_comparison.csv").to_dict(orient="records"),
    }


def get_appointments_dashboard() -> dict[str, Any]:
    appointments = _read_csv(DATA_DIR / "appointments.csv")
    schedule = _read_csv(OUTPUT_DIR / "optimized_appointment_schedule.csv")
    no_show = _read_csv(ML_DIR / "no_show_prediction" / "no_show_predictions.csv")
    if not no_show.empty and "predicted_no_show_probability" in no_show.columns:
        no_show = no_show.sort_values("predicted_no_show_probability", ascending=False)
    if not schedule.empty:
        unassigned = int(schedule["status"].astype(str).str.casefold().isin({"unassigned", "unbooked", "unscheduled"}).sum()) if "status" in schedule.columns else 0
    else:
        unassigned = 0
    return {
        "source_appointment_count": int(len(appointments)),
        "optimized_count": int(len(schedule)),
        "unassigned_count": unassigned,
        "no_show_prediction_count": int(len(no_show)),
        "kpis": _read_csv(OUTPUT_DIR / "appointment_optimization_kpis.csv").to_dict(orient="records"),
        "appointment_schedule": schedule.to_dict(orient="records"),
        # Highest-risk rows only; band counts over all rows are at /no-show/summary.
        "no_show_predictions": no_show.head(500).to_dict(orient="records"),
    }


def get_optimization_dashboard() -> dict[str, Any]:
    return {
        "manifest": _read_manifest(),
        "bed_kpis": _read_csv(OUTPUT_DIR / "bed_allocation_kpis.csv").to_dict(orient="records"),
        "allocations": _read_csv(OUTPUT_DIR / "bed_allocation_simulation.csv").to_dict(orient="records"),
        "emergency_capacity": _read_csv(OUTPUT_DIR / "emergency_capacity_optimization.csv").to_dict(orient="records"),
        "appointment_kpis": _read_csv(OUTPUT_DIR / "appointment_optimization_kpis.csv").to_dict(orient="records"),
        "staff_kpis": _read_csv(OUTPUT_DIR / "staff_coverage_kpis.csv").to_dict(orient="records"),
        "capacity_expansion": _read_csv(OUTPUT_DIR / "capacity_expansion_plan.csv").to_dict(orient="records"),
        "resource_gaps": _read_csv(OUTPUT_DIR / "resource_gap_analysis.csv").to_dict(orient="records"),
        "forecast_backtest": _read_csv(OUTPUT_DIR / "forecast_backtest_summary.csv").to_dict(orient="records"),
        "reliability_scorecard": _read_csv(OUTPUT_DIR / "reliability_scorecard.csv").to_dict(orient="records"),
        "reliability_details": _read_csv(OUTPUT_DIR / "reliability_details.csv").to_dict(orient="records"),
    }
