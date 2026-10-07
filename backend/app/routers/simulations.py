from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.api import SimulationRequest
from backend.app.services.simulation_service import (
    get_simulation,
    get_simulation_runs,
    simulate_scenario,
)

router = APIRouter(prefix="", tags=["simulations"])


@router.post("/simulation/run")
def run_simulation(payload: SimulationRequest) -> dict:
    scenario = simulate_scenario(
        payload.scenario_name,
        payload.admissions_multiplier,
        payload.icu_bed_delta,
        payload.staff_availability_delta,
    )
    return scenario


@router.get("/simulation/runs")
def list_simulations() -> list[dict]:
    return get_simulation_runs()


@router.get("/simulation/{simulation_id}")
def get_simulation_route(simulation_id: str) -> dict:
    simulation = get_simulation(simulation_id)
    if simulation is None:
        raise HTTPException(status_code=404, detail="Simulation not found.")
    return simulation
