from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.api import OptimizationRunRequest
from backend.app.services.pipeline_service import (
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
