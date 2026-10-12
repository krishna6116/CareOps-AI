"""PostgreSQL persistence checks.

Skipped unless CAREOPS_TEST_DATABASE_URL points at a disposable PostgreSQL
database (CI starts one as a service container). Example:

    CAREOPS_TEST_DATABASE_URL=postgresql+psycopg://careops:pw@localhost:5432/careops_test
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

POSTGRES_URL = os.environ.get("CAREOPS_TEST_DATABASE_URL")
# CI sets CAREOPS_REQUIRE_POSTGRES_TESTS=1 so a missing database fails the
# job instead of silently skipping.
REQUIRED = os.environ.get("CAREOPS_REQUIRE_POSTGRES_TESTS") == "1"
if REQUIRED and not POSTGRES_URL:
    raise RuntimeError("CAREOPS_REQUIRE_POSTGRES_TESTS=1 but CAREOPS_TEST_DATABASE_URL is not set.")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL, reason="CAREOPS_TEST_DATABASE_URL is not set; PostgreSQL tests skipped."
)

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "optimization" / "resource_planning" / "outputs"


@pytest.fixture(scope="module")
def engine():
    """Schema created through the documented setup step: alembic upgrade head."""
    environment = {**os.environ, "CAREOPS_DATABASE_URL": POSTGRES_URL}
    subprocess.run(
        [sys.executable, "-m", "alembic", "-c", "backend/alembic.ini", "upgrade", "head"],
        cwd=ROOT, env=environment, check=True,
    )
    engine = create_engine(POSTGRES_URL, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture()
def sessions(engine, monkeypatch: pytest.MonkeyPatch):
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    for target in (
        "backend.app.database.SessionLocal",
        "backend.app.services.ai_assistant.SessionLocal",
        "backend.app.services.dashboard_service.SessionLocal",
        "backend.app.services.simulation_service.SessionLocal",
    ):
        monkeypatch.setattr(target, factory)
    return factory


def test_migrations_create_every_model_table_on_postgresql(engine) -> None:
    from backend.app.database import Base
    from backend.app.models import careops  # noqa: F401

    assert engine.dialect.name == "postgresql"
    tables = set(inspect(engine).get_table_names())
    assert set(Base.metadata.tables) <= tables
    with engine.connect() as connection:
        assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar()


def test_synthetic_records_insert_query_and_roll_back_on_bad_reference(sessions) -> None:
    from backend.app.models.careops import Admission, Bed, Patient

    with sessions() as session:
        session.add_all([
            Patient(patient_id="PGTEST-P1", patient_name="Synthetic Patient", age=60, gender="F"),
            Bed(bed_id="PGTEST-B1", department="Cardiology", bed_type="General", status="Available"),
        ])
        session.flush()
        session.add(Admission(
            admission_id="PGTEST-A1", patient_id="PGTEST-P1", patient_name="Synthetic Patient",
            department="Cardiology", admission_date=date(2026, 10, 1),
            discharge_date=date(2026, 10, 4), los=3, icu_required=0,
            assigned_bed_id="PGTEST-B1", stay_status="completed",
        ))
        session.commit()

    with sessions() as session:
        stored = session.get(Admission, "PGTEST-A1")
        assert stored is not None and stored.los == 3
        session.add(Admission(
            admission_id="PGTEST-A2", patient_id="PGTEST-MISSING", patient_name="Nobody",
            department="Cardiology", admission_date=date(2026, 10, 1),
            discharge_date=date(2026, 10, 2), los=1, icu_required=0,
            assigned_bed_id="PGTEST-B1", stay_status="completed",
        ))
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
        count = session.scalar(
            select(func.count()).select_from(Admission).where(Admission.admission_id.like("PGTEST-%"))
        )
        assert count == 1
        session.delete(stored)
        session.commit()
        session.delete(session.get(Patient, "PGTEST-P1"))
        session.delete(session.get(Bed, "PGTEST-B1"))
        session.commit()


def test_forecasts_are_saved_and_retrieved(sessions) -> None:
    from backend.app.models.careops import AdmissionForecast

    forecast = pd.read_csv(ROOT / "ml" / "admission_forecasting" / "hospital_30_day_admission_forecast.csv")
    run_id = "pgtest-forecast"
    with sessions() as session:
        session.add_all([
            AdmissionForecast(
                run_id=run_id, date=date.fromisoformat(row.date), department=row.department,
                predicted_admissions=float(row.predicted_admissions),
            )
            for row in forecast.itertuples()
        ])
        session.commit()
    with sessions() as session:
        rows = session.scalars(
            select(AdmissionForecast).where(AdmissionForecast.run_id == run_id).order_by(AdmissionForecast.date)
        ).all()
        assert [row.predicted_admissions for row in rows] == forecast["predicted_admissions"].astype(float).tolist()
        for row in rows:
            session.delete(row)
        session.commit()


def test_simulation_and_assistant_audit_survive_an_application_restart(sessions) -> None:
    from backend.app.models.careops import AssistantAudit, SimulationRecord, SimulationResult
    from backend.app.services.ai_assistant import HealthcareAssistant, LLMProvider
    from backend.app.services.assistant_facts import answer_question
    from backend.app.services.simulation_service import simulate_scenario

    result = simulate_scenario("pgtest-scenario", 1.2, -3, -0.1)
    question = "How many beds are forecast to be available in 7 days?"
    HealthcareAssistant(LLMProvider(provider="deterministic")).answer_audited(
        question,
        {"scenario_as_of_date": "pgtest", "grounded_answer": answer_question(question).to_dict()},
        ["bed_availability_forecast.csv"],
    )

    restarted = create_engine(POSTGRES_URL)  # a fresh process would build a new engine
    try:
        with sessionmaker(bind=restarted)() as session:
            record = session.get(SimulationRecord, result["simulation_id"])
            assert record is not None
            assert record.assumptions == result["assumptions"]
            assert record.simulated_metrics["overflow_bed_days"] == result["simulated_metrics"]["overflow_bed_days"]
            results = session.scalar(
                select(func.count()).select_from(SimulationResult)
                .where(SimulationResult.simulation_id == result["simulation_id"])
            )
            assert results == len(result["baseline_metrics"])
            audit = session.scalars(
                select(AssistantAudit).where(AssistantAudit.run_id == "pgtest")
            ).all()
            assert audit and audit[-1].question == question and audit[-1].response_status == "success"
    finally:
        restarted.dispose()


def test_pipeline_handoff_loads_every_output_atomically_and_idempotently(engine) -> None:
    from optimization.postgresql_integration import ARTIFACTS, SCHEMA, load_pipeline_to_postgres

    manifest = json.loads((OUTPUTS / "portfolio_run_manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["scenario_as_of_date"]

    def counts() -> dict[str, int]:
        with engine.connect() as connection:
            return {
                table: connection.execute(
                    text(f'SELECT count(*) FROM "{SCHEMA}"."{table}" WHERE run_id = :run_id'),
                    {"run_id": run_id},
                ).scalar_one()
                for table in ARTIFACTS
            }

    summary = load_pipeline_to_postgres(ROOT, manifest, POSTGRES_URL)
    assert summary["status"] == "loaded"
    assert counts() == summary["tables"]
    assert summary["tables"]["optimization_bed_assignment_requests"] == len(
        pd.read_csv(OUTPUTS / "bed_assignment_requests.csv")
    )

    load_pipeline_to_postgres(ROOT, manifest, POSTGRES_URL)
    assert counts() == summary["tables"], "a reload must replace, not duplicate, the run"
    with engine.connect() as connection:
        stored = connection.execute(
            text(f'SELECT pipeline_status FROM "{SCHEMA}".pipeline_runs WHERE run_id = :run_id'),
            {"run_id": run_id},
        ).scalar_one()
    assert stored == manifest["pipeline_status"]


def test_api_reports_postgresql_health(engine, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    from backend.app.main import app

    monkeypatch.setattr("backend.app.database.engine", engine)
    payload = TestClient(app).get("/health").json()
    assert payload["database"]["status"] == "ok"
    assert payload["database"]["engine"] == "postgresql"
