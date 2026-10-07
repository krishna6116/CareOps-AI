from __future__ import annotations

from fastapi import APIRouter, Query

from backend.app.services.pipeline_service import (
    get_assistant_audit_runs,
    get_audit_runs,
)

router = APIRouter(prefix="", tags=["audit"])


@router.get("/audit/runs")
def list_audit_runs() -> list[dict]:
    return get_audit_runs()


@router.get("/audit/assistant")
def list_assistant_audit_runs(
    limit: int = Query(default=100, ge=1, le=1000),
) -> list[dict]:
    return get_assistant_audit_runs(limit)
