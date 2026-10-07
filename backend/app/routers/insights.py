from __future__ import annotations

import re
from math import isfinite
from statistics import fmean
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from backend.app.schemas.api import AssistantRequest, SimulationRequest
from backend.app.services.ai_assistant import AssistantProviderError, HealthcareAssistant
from backend.app.services.dashboard_service import (
    get_appointments_dashboard,
    get_bed_management,
    get_optimization_dashboard,
    get_overview,
    get_staff_dashboard,
)
from backend.app.services.pipeline_service import (
    get_admission_forecast,
    get_bed_availability,
    get_historical_operations,
    get_icu_forecast,
    get_los_predictions,
    get_model_metrics,
    get_no_show_predictions,
    get_occupancy_forecast,
)
from backend.app.services.simulation_service import (
    get_simulation_runs,
    simulate_scenario,
)

router = APIRouter(prefix="", tags=["dashboard and insights"])


def _scenario_from_question(question: str) -> dict[str, Any] | None:
    lower = question.casefold()
    if not any(term in lower for term in ("what if", "what happens if", "simulate")):
        return None

    scenario: dict[str, Any] = {}
    percentage = re.search(r"(\d+(?:\.\d+)?)\s*%", lower)
    if percentage and "admission" in lower and any(
        term in lower for term in ("increas", "higher", "more", "+")
    ):
        scenario["admissions_multiplier"] = 1 + float(percentage.group(1)) / 100
    elif percentage and "admission" in lower and any(
        term in lower for term in ("decreas", "lower", "fewer", "less")
    ):
        scenario["admissions_multiplier"] = 1 - float(percentage.group(1)) / 100

    icu_change = re.search(
        r"icu.{0,30}?(?:decreas\w*|reduc\w*|drop\w*|remov\w*)\D{0,12}(\d+)",
        lower,
    ) or re.search(
        r"(?:decreas\w*|reduc\w*|drop\w*|remov\w*).{0,30}?"
        r"icu.{0,20}?(?:by\s+)?(\d+)",
        lower,
    ) or re.search(r"(\d+)\s+fewer\s+icu\s+beds?", lower)
    if icu_change:
        scenario["icu_bed_delta"] = -int(icu_change.group(1))

    staff_change = re.search(
        r"staff(?:\s+availability)?.{0,40}?"
        r"(?:decreas\w*|reduc\w*|drop\w*|less)\D{0,12}"
        r"(\d+(?:\.\d+)?)\s*%",
        lower,
    )
    if staff_change:
        scenario["staff_availability_delta"] = -float(staff_change.group(1)) / 100

    return scenario or None


@router.get("/dashboard/overview")
def dashboard_overview() -> dict[str, Any]:
    return get_overview()


@router.get("/dashboard/beds")
def dashboard_beds() -> list[dict[str, Any]]:
    return get_bed_management()


@router.get("/dashboard/staff")
def dashboard_staff() -> dict[str, Any]:
    return get_staff_dashboard()


@router.get("/dashboard/appointments")
def dashboard_appointments() -> dict[str, Any]:
    return get_appointments_dashboard()


@router.get("/dashboard/optimization")
def dashboard_optimization() -> dict[str, Any]:
    return get_optimization_dashboard()


@router.get("/forecast/icu")
def icu_forecast() -> list[dict[str, Any]]:
    return get_icu_forecast()


@router.get("/forecast/history")
def historical_operations(
    days: int = Query(default=90, ge=1, le=1008),
) -> list[dict[str, Any]]:
    return get_historical_operations(days)


@router.get("/los/predictions")
def los_predictions(
    limit: int = Query(default=500, ge=1, le=5000),
) -> list[dict[str, Any]]:
    return get_los_predictions()[:limit]


@router.get("/no-show/predictions")
def no_show_predictions(
    limit: int = Query(default=500, ge=1, le=100000),
) -> list[dict[str, Any]]:
    return get_no_show_predictions(limit)


@router.get("/models/metrics")
def model_metrics() -> dict[str, Any]:
    return get_model_metrics()


@router.post("/assistant/ask")
def ask_assistant(payload: AssistantRequest) -> dict[str, Any]:
    overview = get_overview()
    optimization = get_optimization_dashboard()
    admissions = optimization["manifest"].get("scenario_as_of_date")
    if not admissions:
        raise HTTPException(
            status_code=503,
            detail="No completed portfolio run is available to explain.",
        )
    scenario_request = _scenario_from_question(payload.question)
    sources = [
        "portfolio_run_manifest.json",
        "current synthetic bed inventory",
        "generated forecast and optimization outputs",
    ]
    if scenario_request:
        try:
            request = SimulationRequest(
                scenario_name=f"assistant_{uuid4().hex[:12]}",
                **scenario_request,
            )
        except ValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail="Requested scenario is outside the supported simulation limits.",
            ) from exc
        simulate_scenario(
            request.scenario_name,
            request.admissions_multiplier,
            request.icu_bed_delta,
            request.staff_availability_delta,
        )
        sources.append("OR-Tools what-if simulation")

    los_predictions = get_los_predictions()
    predicted_los = []
    for row in los_predictions:
        try:
            value = float(row["predicted_los_days"])
        except (KeyError, TypeError, ValueError):
            continue
        if isfinite(value):
            predicted_los.append(value)
    los_summary = {
        "prediction_count": len(predicted_los),
        "mean_predicted_los_days": round(fmean(predicted_los), 2),
    } if predicted_los else {}
    context = {
        **overview,
        "scenario_as_of_date": admissions,
        "admission_forecast": get_admission_forecast(),
        "occupancy_forecast": get_occupancy_forecast(),
        "los_prediction_summary": los_summary,
        "bed_availability": get_bed_availability()[:100],
        "bed_kpis": optimization["bed_kpis"],
        "icu_forecast": get_icu_forecast(),
        "recommendations": overview.get("operational_recommendations", []),
        "alerts": overview.get("alerts", 0),
        "model_metrics": get_model_metrics(),
        "simulations": get_simulation_runs(limit=20),
    }
    assistant = HealthcareAssistant()
    try:
        answer = assistant.answer_audited(
            payload.question,
            context,
            sources,
        )
    except AssistantProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return {
        "answer": answer,
        "source_run_id": admissions,
        "sources": sources,
        "llm_configured": assistant.provider.is_configured,
        "assistant_provider": assistant.provider.provider,
    }
