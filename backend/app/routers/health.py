from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter

from backend.app.config import settings
from backend.app.database import check_database_health

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
