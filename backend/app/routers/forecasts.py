from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.schemas.api import ForecastRequest, LOSRequest
from backend.app.services.pipeline_service import (
    get_admission_forecast,
    get_occupancy_forecast,
    get_los_predictions,
)

router = APIRouter(prefix="", tags=["forecasts"])


@router.post("/forecast/admissions")
def forecast_admissions(payload: ForecastRequest) -> list[dict[str, Any]]:
    return get_admission_forecast(payload.as_of_date.isoformat() if payload.as_of_date else None)


@router.get("/forecast/admissions")
def get_admissions_forecast() -> list[dict[str, Any]]:
    return get_admission_forecast()


@router.post("/forecast/occupancy")
def forecast_occupancy(payload: ForecastRequest) -> list[dict[str, Any]]:
    return get_occupancy_forecast(payload.as_of_date.isoformat() if payload.as_of_date else None)


@router.get("/forecast/occupancy")
def get_occupancy_forecast_route() -> list[dict[str, Any]]:
    return get_occupancy_forecast()


@router.post("/los/predict")
def predict_los(payload: LOSRequest) -> dict[str, Any]:
    predictions = get_los_predictions()
    if payload.patient_id:
        predictions = [
            row for row in predictions
            if str(row.get("patient_id", "")) == payload.patient_id
        ]
    if payload.department:
        predictions = [
            row for row in predictions
            if str(row.get("department", "")).casefold() == payload.department.casefold()
        ]
    if not predictions:
        raise HTTPException(
            status_code=404,
            detail="No saved LOS prediction matches the requested patient or department.",
        )
    return {
        "request": payload.model_dump(exclude_none=True),
        "predictions": predictions[:10],
        "prediction_count": len(predictions),
        "source": "saved synthetic LOS model output filtered by the requested selector",
    }
