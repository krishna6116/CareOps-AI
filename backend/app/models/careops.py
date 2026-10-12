from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, Column, Date, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from backend.app.database import Base


def utc_now() -> datetime:
    """Naive UTC timestamp, matching the existing timezone-less columns."""
    return datetime.now(UTC).replace(tzinfo=None)


class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(String(64), primary_key=True)
    patient_name = Column(String(160), nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=utc_now)


class Bed(Base):
    __tablename__ = "beds"

    bed_id = Column(String(64), primary_key=True)
    department = Column(String(128), nullable=False)
    bed_type = Column(String(64), nullable=False)
    status = Column(String(32), nullable=False)
    isolation_capable = Column(Boolean, default=False)
    emergency_reserve = Column(Boolean, default=False)
    available_for_emergency = Column(Boolean, default=False)
    created_at = Column(DateTime, default=utc_now)


class Staff(Base):
    __tablename__ = "staff"

    staff_id = Column(String(64), primary_key=True)
    department = Column(String(128), nullable=False)
    role = Column(String(64), nullable=False)
    availability = Column(String(32), nullable=False)
    max_hours = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now)


class Admission(Base):
    __tablename__ = "admissions"

    admission_id = Column(String(64), primary_key=True)
    patient_id = Column(String(64), ForeignKey("patients.patient_id"), nullable=False)
    patient_name = Column(String(160), nullable=False)
    department = Column(String(128), nullable=False)
    admission_date = Column(Date, nullable=False)
    discharge_date = Column(Date, nullable=False)
    los = Column(Integer, nullable=False)
    icu_required = Column(Integer, nullable=False)
    assigned_bed_id = Column(String(64), ForeignKey("beds.bed_id"), nullable=False)
    stay_status = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=utc_now)


class DischargeEvent(Base):
    __tablename__ = "discharges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    admission_id = Column(String(64), ForeignKey("admissions.admission_id"), nullable=False)
    department = Column(String(128), nullable=False)
    discharge_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=utc_now)


class Appointment(Base):
    __tablename__ = "appointments"

    appointment_id = Column(String(64), primary_key=True)
    patient_id = Column(String(64), ForeignKey("patients.patient_id"), nullable=False)
    doctor_id = Column(String(64), ForeignKey("staff.staff_id"), nullable=False)
    department = Column(String(128), nullable=False)
    appointment_date = Column(Date, nullable=False)
    appointment_time = Column(String(16), nullable=False)
    duration = Column(Integer, nullable=False)
    no_show = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=utc_now)


class Shift(Base):
    __tablename__ = "shifts"

    shift_id = Column(String(64), primary_key=True)
    staff_id = Column(String(64), ForeignKey("staff.staff_id"), nullable=False)
    duty_date = Column(Date, nullable=False)
    shift_name = Column(String(32), nullable=False)
    hours = Column(Float, nullable=False)
    assigned = Column(String(8), nullable=False)
    created_at = Column(DateTime, default=utc_now)


class AdmissionForecast(Base):
    __tablename__ = "admission_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    date = Column(Date, nullable=False)
    department = Column(String(128), nullable=False)
    predicted_admissions = Column(Float, nullable=False)
    created_at = Column(DateTime, default=utc_now)


class OccupancyForecast(Base):
    __tablename__ = "occupancy_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    date = Column(Date, nullable=False)
    forecast_occupied_beds = Column(Float, nullable=False)
    utilization = Column(Float, nullable=True)
    available_capacity = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class LOSPrediction(Base):
    __tablename__ = "los_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    admission_id = Column(String(64), ForeignKey("admissions.admission_id"), nullable=False)
    expected_los_days = Column(Float, nullable=False)
    prediction_interval_low = Column(Float, nullable=True)
    prediction_interval_high = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class BedAvailability(Base):
    __tablename__ = "bed_availability"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    department = Column(String(128), nullable=False)
    date = Column(Date, nullable=False)
    total_beds = Column(Integer, nullable=False)
    occupied_beds = Column(Integer, nullable=False)
    available_beds = Column(Integer, nullable=False)
    reserved_beds = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=utc_now)


class BedAllocation(Base):
    __tablename__ = "bed_allocations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    department = Column(String(128), nullable=False)
    allocation_status = Column(String(32), nullable=False)
    allocated_beds = Column(Integer, nullable=False)
    overflow_bed_days = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    run_id = Column(String(64), primary_key=True)
    scenario = Column(String(128), nullable=False)
    started_at = Column(DateTime, default=utc_now)
    completed_at = Column(DateTime, nullable=True)
    optimizer_status = Column(String(64), nullable=False)
    kpis = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class AlertLog(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    severity = Column(String(32), nullable=False)
    category = Column(String(64), nullable=False)
    message = Column(Text, nullable=False)
    source = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=utc_now)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    department = Column(String(128), nullable=True)
    title = Column(String(200), nullable=False)
    detail = Column(Text, nullable=False)
    priority = Column(String(32), nullable=False)
    created_at = Column(DateTime, default=utc_now)


class SimulationRecord(Base):
    __tablename__ = "simulations"

    id = Column(String(64), primary_key=True)
    scenario = Column(String(128), nullable=False)
    assumptions = Column(JSON, nullable=False)
    baseline_metrics = Column(JSON, nullable=False)
    simulated_metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=utc_now)


class SimulationResult(Base):
    __tablename__ = "simulation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String(64), ForeignKey("simulations.id"), nullable=False)
    metric_name = Column(String(128), nullable=False)
    baseline_value = Column(Float, nullable=True)
    simulated_value = Column(Float, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(128), nullable=False)
    run_id = Column(String(64), nullable=False)
    version = Column(String(128), nullable=False)
    metrics = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=utc_now)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False)
    check_name = Column(String(128), nullable=False)
    status = Column(String(32), nullable=False)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)


class AssistantAudit(Base):
    __tablename__ = "assistant_audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=True)
    question = Column(Text, nullable=False)
    provider = Column(String(64), nullable=False)
    model = Column(String(128), nullable=False)
    response_status = Column(String(32), nullable=False)
    sources = Column(JSON, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)
