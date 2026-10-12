from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "optimization" / "resource_planning" / "outputs"
ML_ROOT = PROJECT_ROOT / "ml"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def read_manifest() -> dict:
    manifest_path = OUTPUT_DIR / "portfolio_run_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def get_patients() -> list[dict]:
    return _read_csv(SYNTHETIC_DIR / "patients.csv").head(100).to_dict(orient="records")


def get_admissions() -> list[dict]:
    return _read_csv(SYNTHETIC_DIR / "admissions.csv").head(100).to_dict(orient="records")


def get_beds() -> list[dict]:
    return _read_csv(SYNTHETIC_DIR / "beds.csv").head(200).to_dict(orient="records")


def get_bed_availability() -> list[dict]:
    path = OUTPUT_DIR / "bed_availability_forecast.csv"
    if path.exists():
        return _read_csv(path).to_dict(orient="records")
    return _read_csv(SYNTHETIC_DIR / "beds.csv").rename(columns={"status": "availability"}).to_dict(orient="records")


def get_admission_forecast(as_of_date: str | None = None) -> list[dict]:
    path = ML_ROOT / "admission_forecasting" / "admission_30_day_forecast.csv"
    if not path.exists():
        return []
    frame = _read_csv(path)
    if as_of_date:
        frame = frame[pd.to_datetime(frame["date"], errors="coerce") >= pd.Timestamp(as_of_date)]
    return frame.to_dict(orient="records")


def get_occupancy_forecast(as_of_date: str | None = None) -> list[dict]:
    path = ML_ROOT / "occupancy_forecasting" / "occupancy_future_forecast.csv"
    if not path.exists():
        return []
    frame = _read_csv(path)
    if as_of_date:
        frame = frame[pd.to_datetime(frame["date"], errors="coerce") >= pd.Timestamp(as_of_date)]
    return frame.to_dict(orient="records")


def get_icu_forecast() -> list[dict[str, Any]]:
    path = ML_ROOT / "icu_demand_forecasting" / "icu_future_forecast.csv"
    if not path.exists():
        return []
    return _read_csv(path).to_dict(orient="records")


def get_historical_operations(days: int = 90) -> list[dict[str, Any]]:
    path = SYNTHETIC_DIR / "daily_operations.csv"
    if not path.exists():
        return []
    frame = _read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    if "department" in frame.columns:
        numeric_columns = frame.select_dtypes(include="number").columns.tolist()
        frame = frame.groupby("date", as_index=False)[numeric_columns].sum()
    frame = frame.sort_values("date").tail(days)
    frame["date"] = frame["date"].dt.date.astype(str)
    return frame.to_dict(orient="records")


def get_los_predictions() -> list[dict]:
    path = ML_ROOT / "los_prediction" / "los_predictions.csv"
    if not path.exists():
        return []
    return _read_csv(path).to_dict(orient="records")


def get_no_show_predictions(limit: int = 500) -> list[dict[str, Any]]:
    path = ML_ROOT / "no_show_prediction" / "no_show_predictions.csv"
    if not path.exists():
        return []
    frame = _read_csv(path)
    if "predicted_no_show_probability" in frame.columns:
        frame = frame.sort_values("predicted_no_show_probability", ascending=False)
    return frame.head(limit).to_dict(orient="records")


def get_staff_schedule(limit: int = 1000) -> list[dict[str, Any]]:
    path = OUTPUT_DIR / "optimized_staff_schedule.csv"
    if not path.exists():
        return []
    return _read_csv(path).head(limit).to_dict(orient="records")


def get_appointment_schedule(limit: int = 1000) -> list[dict[str, Any]]:
    path = OUTPUT_DIR / "optimized_appointment_schedule.csv"
    if not path.exists():
        return []
    return _read_csv(path).head(limit).to_dict(orient="records")


def get_model_metrics() -> dict[str, Any]:
    artifacts = {
        "admissions": ML_ROOT / "admission_forecasting" / "model_summary.json",
        "occupancy": ML_ROOT / "occupancy_forecasting" / "run_metadata.json",
        "icu": ML_ROOT / "icu_demand_forecasting" / "icu_demand_metrics.json",
        "los": ML_ROOT / "los_prediction" / "los_metrics.json",
        "no_show": ML_ROOT / "no_show_prediction" / "no_show_metrics.json",
    }
    metrics: dict[str, Any] = {}
    for model_name, path in artifacts.items():
        if path.is_file():
            metrics[model_name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def get_audit_runs() -> list[dict]:
    path = OUTPUT_DIR / "data_quality_audit.csv"
    if not path.exists():
        return []
    return _read_csv(path).head(200).to_dict(orient="records")


def get_assistant_audit_runs(limit: int = 100) -> list[dict[str, Any]]:
    from backend.app.database import SessionLocal
    from backend.app.models.careops import AssistantAudit

    with SessionLocal() as session:
        records = (
            session.query(AssistantAudit)
            .order_by(AssistantAudit.created_at.desc(), AssistantAudit.id.desc())
            .limit(limit)
            .all()
        )
    return [
        {
            "id": row.id,
            "run_id": row.run_id,
            "question": row.question,
            "provider": row.provider,
            "model": row.model,
            "response_status": row.response_status,
            "sources": row.sources,
            "latency_ms": row.latency_ms,
            "error": row.error,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in records
    ]


def run_portfolio_pipeline(as_of_date: str | None = None, postgres: bool = False) -> dict:
    from run_portfolio import run_pipeline

    scenario_date = date.fromisoformat(as_of_date) if as_of_date else date.today()
    run_pipeline(scenario_date, postgres=postgres)
    return read_manifest()


def get_optimization_runs() -> list[dict]:
    manifest = read_manifest()
    if not manifest:
        return []
    return [manifest]


def get_optimization_run(run_id: str) -> dict:
    manifest = read_manifest()
    if not manifest or str(manifest.get("scenario_as_of_date")) != run_id:
        return {"run_id": run_id, "status": "not_found"}
    return {"run_id": run_id, **manifest}


def get_optimization_allocations(run_id: str) -> list[dict]:
    manifest = read_manifest()
    if not manifest or str(manifest.get("scenario_as_of_date")) != run_id:
        return []
    path = OUTPUT_DIR / "bed_allocation_simulation.csv"
    if not path.exists():
        return []
    return _read_csv(path).to_dict(orient="records")


def get_bed_assignment_summary() -> dict[str, Any]:
    """Request-level replay: summary, policy KPIs, daily solver log, sensitivity."""
    path = OUTPUT_DIR / "bed_assignment_summary.json"
    if not path.exists():
        return {}
    return {
        **json.loads(path.read_text(encoding="utf-8")),
        "daily": _read_csv(OUTPUT_DIR / "bed_assignment_daily.csv").to_dict(orient="records"),
        "sensitivity": _read_csv(OUTPUT_DIR / "bed_assignment_sensitivity.csv").to_dict(orient="records"),
    }


def get_bed_assignment_requests(
    policy: str = "optimized",
    status: str | None = None,
    planning_date: str | None = None,
    department: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> dict[str, Any]:
    frame = _read_csv(OUTPUT_DIR / "bed_assignment_requests.csv")
    if frame.empty:
        return {"total": 0, "offset": offset, "limit": limit, "rows": []}
    if status:
        frame = frame[frame[f"{policy}_status"].str.casefold() == status.casefold()]
    if planning_date:
        frame = frame[frame["planning_date"] == planning_date]
    if department:
        frame = frame[frame["department"].str.casefold() == department.casefold()]
    page = frame.iloc[offset:offset + limit]
    return {
        "total": int(len(frame)),
        "offset": offset,
        "limit": limit,
        "rows": page.astype(object).where(page.notna(), None).to_dict(orient="records"),
    }


def get_model_evaluation() -> dict[str, Any]:
    """Published model vs baselines, interval coverage and alert-threshold scores.

    Uses the same summary as docs/MODEL_METRICS.md so the dashboard and the
    report cannot disagree.
    """
    from ml import model_metrics_report as report

    try:
        data = report.load_outputs()
    except FileNotFoundError:
        return {}

    def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return frame.astype(object).where(frame.notna(), None).to_dict(orient="records")

    summary = report.summary_rows(data).rename(columns=lambda column: (
        column.casefold().replace(" ", "_")
    ))
    summary["learned_beats_baseline"] = summary["learned_beats_baseline"].eq("yes")
    coverage = []
    for target, key in (
        ("admissions", "admission_coverage"),
        ("occupied_beds", "occupancy_coverage"),
        ("icu_admissions", "icu_coverage"),
    ):
        frame = data[key]
        # Only the method used for the published forecasts, as in the report.
        frame = frame[frame["method"].eq("sequential_published")].copy()
        frame.insert(0, "target", target)
        coverage.extend(records(frame))
    return {
        "scenario_as_of_date": data["manifest"].get("scenario_as_of_date"),
        "summary": records(summary),
        "interval_coverage": coverage,
        "occupancy_risk_thresholds": records(data["occupancy_risk"]),
        "report": "docs/MODEL_METRICS.md",
    }


NO_SHOW_BANDS = (("below_0_2", 0.0, 0.2), ("0_2_to_0_5", 0.2, 0.5), ("0_5_and_above", 0.5, 1.000001))


def get_no_show_summary() -> dict[str, Any]:
    """Risk-band counts over every saved prediction, so clients need not download them."""
    path = ML_ROOT / "no_show_prediction" / "no_show_predictions.csv"
    if not path.exists():
        return {}
    frame = _read_csv(path)
    probability = pd.to_numeric(frame["predicted_no_show_probability"], errors="coerce")
    bands = {
        name: int(((probability >= low) & (probability < high)).sum())
        for name, low, high in NO_SHOW_BANDS
    }
    observed = pd.to_numeric(frame.get("actual_no_show"), errors="coerce")
    return {
        "prediction_count": int(probability.notna().sum()),
        "risk_bands": bands,
        "mean_predicted_probability": round(float(probability.mean()), 4),
        "observed_no_show_rate": round(float(observed.mean()), 4) if observed is not None else None,
        "window": [str(frame["appointment_datetime"].min()), str(frame["appointment_datetime"].max())],
    }
