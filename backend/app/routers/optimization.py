from __future__ import annotations

from datetime import date
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.api import OptimizationRunRequest
from backend.app.services.pipeline_service import (
    get_bed_assignment_requests,
    get_bed_assignment_summary,
    get_optimization_allocations,
    get_optimization_run,
    get_optimization_runs,
    run_portfolio_pipeline,
)

router = APIRouter(prefix="", tags=["optimization"])


@router.post("/optimization/run")
def run_optimization(payload: OptimizationRunRequest) -> dict:
    try:
        result = run_portfolio_pipeline(
            payload.as_of_date.isoformat() if payload.as_of_date else None,
            payload.postgres,
        )
    except Exception as exc:  # pragma: no cover - integration route
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


@router.get("/optimization/runs")
def list_optimization_runs() -> list[dict]:
    return get_optimization_runs()


@router.get("/optimization/runs/{run_id}")
def read_optimization_run(run_id: str) -> dict:
    run = get_optimization_run(run_id)
    if run.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


@router.get("/optimization/runs/{run_id}/allocations")
def read_optimization_allocations(run_id: str) -> list[dict]:
    allocations = get_optimization_allocations(run_id)
    if not allocations:
        raise HTTPException(status_code=404, detail="Allocation data not found for this run.")
    return allocations


@router.get("/optimization/bed-assignments")
def bed_assignment_summary() -> dict[str, Any]:
    summary = get_bed_assignment_summary()
    if not summary:
        raise HTTPException(
            status_code=404,
            detail="No request-level bed assignment run is available. Run run_portfolio.py.",
        )
    return summary


@router.get("/optimization/bed-assignments/requests")
def bed_assignment_requests(
    policy: Literal["baseline", "optimized"] = "optimized",
    status: Literal["ASSIGNED", "UNASSIGNED"] | None = None,
    planning_date: date | None = None,
    department: str | None = Query(default=None, max_length=64),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    return get_bed_assignment_requests(
        policy=policy,
        status=status,
        planning_date=planning_date.isoformat() if planning_date else None,
        department=department,
        offset=offset,
        limit=limit,
    )
