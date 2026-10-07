from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    database: str
    timestamp: datetime


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of_date: date | None = None


class LOSRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str | None = None
    department: str | None = None

    @model_validator(mode="after")
    def require_prediction_filter(self) -> LOSRequest:
        if not self.patient_id and not self.department:
            raise ValueError("Provide a patient_id or department to select saved LOS predictions.")
        return self


class OptimizationRunRequest(BaseModel):
    as_of_date: date | None = None
    postgres: bool = False


class SimulationRequest(BaseModel):
    scenario_name: str = "baseline"
    admissions_multiplier: float = Field(default=1.0, ge=0, le=3)
    icu_bed_delta: int = Field(default=0, ge=-20, le=20)
    staff_availability_delta: float = Field(default=0.0, ge=-1, le=1)


class AssistantRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class AlertCreate(BaseModel):
    severity: Literal["low", "medium", "high", "critical"]
    category: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000)
    source: str | None = Field(default=None, max_length=128)


class RecommendationRequest(BaseModel):
    department: str | None = None


class GenericResponse(BaseModel):
    status: str
    data: dict[str, Any] | list[Any] | None = None
    message: str | None = None
