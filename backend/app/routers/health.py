from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.database import check_database_health
from backend.app.services.pipeline_service import read_manifest

router = APIRouter(prefix="", tags=["health"])


@router.get("/health")
def health_check() -> dict[str, object]:
    db_status = check_database_health()
    database: dict[str, object] = {
        "status": db_status.get("status", "error"),
        "engine": db_status.get("database", "unknown"),
    }
    if db_status.get("status") != "ok":
        database["error"] = "Database connection failed."
    return {
        "status": "ok" if db_status.get("status") == "ok" else "degraded",
        "app": settings.project_name,
        "version": settings.app_version,
        "database": database,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


@router.get("/ready")
def readiness_check() -> JSONResponse:
    """Ready to serve results: database reachable and a completed pipeline run present."""
    database = check_database_health().get("status") == "ok"
    manifest = read_manifest()
    checks = {
        "database": "ok" if database else "unavailable",
        "pipeline_run": manifest.get("scenario_as_of_date") or "missing",
    }
    ready = database and bool(manifest.get("scenario_as_of_date"))
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "not_ready", "checks": checks},
    )


@router.get("/meta")
def application_metadata() -> dict[str, object]:
    manifest = read_manifest()
    return {
        "app": settings.project_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "data_scope": manifest.get("data_scope", "fully_synthetic_portfolio_scenario"),
        "scenario_as_of_date": manifest.get("scenario_as_of_date"),
        "forecast_horizon_days": manifest.get("forecast_horizon_days"),
        "pipeline_status": manifest.get("pipeline_status"),
        "assistant_provider": settings.llm_provider,
        "clinical_use": False,
    }
