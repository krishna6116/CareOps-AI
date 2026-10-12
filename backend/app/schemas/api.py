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


class DatasetCatalogResponse(BaseModel):
    dataset_count: int
    datasets: list[dict[str, Any]]


class DatasetPageResponse(BaseModel):
    dataset: dict[str, Any]
    offset: int
    limit: int
    returned_rows: int
    rows: list[dict[str, Any]]


class DatasetValidationResponse(BaseModel):
    dataset_count: int
    datasets: list[dict[str, Any]]


class OperationsAnalyticsResponse(BaseModel):
    filters: dict[str, Any]
    summary: dict[str, Any]
    daily: list[dict[str, Any]]
    departments: list[dict[str, Any]]


class DepartmentsResponse(BaseModel):
    departments: list[str]


# ---------------------------------------------------------------- responses
# Typed response contracts for the main routes. Row-level tables stay
# ``dict[str, Any]`` because their columns follow the pipeline outputs.

class DatabaseStatus(BaseModel):
    status: Literal["ok", "error"]
    engine: str
    error: str | None = None


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded"]
    app: str
    version: str
    database: DatabaseStatus
    timestamp: str


class AppMetadata(BaseModel):
    app: str
    version: str
    environment: str
    data_scope: str
    scenario_as_of_date: str | None
    forecast_horizon_days: int | None
    pipeline_status: str | None
    assistant_provider: str
    clinical_use: Literal[False]


class OverviewResponse(BaseModel):
    total_beds: int
    occupied_beds: int
    available_beds: int
    unavailable_beds: int
    utilization: float
    predicted_admissions: int
    predicted_occupancy: int | None
    icu_total_beds: int
    icu_occupied_beds: int
    icu_available_beds: int
    emergency_total_beds: int
    emergency_available_beds: int
    alerts: int
    optimization_status: str
    operational_recommendations: list[dict[str, Any]]
    scenario: str
    latest_forecast_date: str | None
    forecast_horizon_days: int
    latest_optimization_run: str | None


class EvaluationSummaryRow(BaseModel):
    target: str
    evaluation: str
    metric: str
    published_model: str
    published: str
    best_baseline: str
    best_learned: str
    learned_beats_baseline: bool


class ModelEvaluationResponse(BaseModel):
    scenario_as_of_date: str | None
    summary: list[EvaluationSummaryRow]
    interval_coverage: list[dict[str, Any]]
    occupancy_risk_thresholds: list[dict[str, Any]]
    report: str


class NoShowSummary(BaseModel):
    prediction_count: int
    risk_bands: dict[str, int]
    mean_predicted_probability: float
    observed_no_show_rate: float | None
    window: list[str]


class BedAssignmentKpis(BaseModel):
    model_config = ConfigDict(extra="allow")

    policy: Literal["baseline", "optimized"]
    requests: int
    assigned: int
    unassigned: int
    assignment_rate: float
    high_severity_assignment_rate: float | None
    off_service_placements: int
    compatibility_violations: int
    staffing_exceedances: int


class BedAssignmentSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    as_of_date: str
    window: list[str]
    baseline_policy: str
    objective: dict[str, int]
    constraints: dict[str, Any]
    solver_statuses: dict[str, int]
    kpis: list[BedAssignmentKpis]
    reason_definitions: dict[str, str]
    limitations: list[str]
    daily: list[dict[str, Any]]
    sensitivity: list[dict[str, Any]]


class BedAssignmentPage(BaseModel):
    total: int
    offset: int
    limit: int
    rows: list[dict[str, Any]]


class SimulationResponse(BaseModel):
    simulation_id: str
    scenario_name: str
    assumptions: dict[str, float]
    baseline_metrics: dict[str, float | int]
    simulated_metrics: dict[str, float | int]
    operational_recommendation: str
    timestamp: str
    source_run_id: str
    solver: str
    scope_note: str


class EvidenceItem(BaseModel):
    label: str
    value: Any
    kind: Literal["observed", "forecast", "model_metric", "simulated"]
    source: str


class AssistantResponse(BaseModel):
    answer: str
    intent: str
    evidence: list[EvidenceItem]
    limitations: list[str]
    source_run_id: str
    evidence_run_id: str | None
    sources: list[str]
    simulation_ran: bool
    llm_configured: bool
    assistant_provider: str
    fallback_reason: str | None
