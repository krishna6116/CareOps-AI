from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import URL, create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.database import Base
from backend.app.main import app
from backend.app.services.ai_assistant import (
    AssistantProviderError,
    LLMProvider,
)

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    engine = create_engine(
        URL.create("sqlite", database=str(tmp_path / "api-test.db")),
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    test_sessions = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr("backend.app.database.engine", engine)
    monkeypatch.setattr("backend.app.database.SessionLocal", test_sessions)
    monkeypatch.setattr("backend.app.services.ai_assistant.SessionLocal", test_sessions)
    monkeypatch.setattr("backend.app.services.dashboard_service.SessionLocal", test_sessions)
    monkeypatch.setattr("backend.app.services.simulation_service.SessionLocal", test_sessions)
    yield
    engine.dispose()


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["app"] == "CareOps-AI"
    assert "database_url" not in payload["database"]
    assert "error" not in payload["database"] or payload["database"]["error"] == "Database connection failed."
    browser_origin = client.get(
        "/health",
        headers={"Origin": "http://127.0.0.1:5173"},
    )
    assert browser_origin.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_health_endpoint_redacts_database_credentials_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backend.app.routers.health.check_database_health",
        lambda: {
            "status": "error",
            "database_url": "postgresql://private:secret@db.example/careops",
            "error": "password=secret",
        },
    )

    payload = client.get("/health").json()

    assert payload["status"] == "degraded"
    assert payload["database"] == {
        "status": "error",
        "engine": "unknown",
        "error": "Database connection failed.",
    }
    assert "secret" not in str(payload)


def test_patients_endpoint() -> None:
    response = client.get("/patients")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) > 0


def test_alerts_are_persisted_and_reflected() -> None:
    created = client.post(
        "/alerts",
        json={
            "severity": "medium",
            "category": "test",
            "message": "Persisted API test alert",
            "source": "test_api",
        },
    )
    assert created.status_code == 200
    alert = created.json()["alert"]
    assert alert["status"] == "open"
    assert alert["id"] is not None
    listed = client.get("/alerts")
    assert listed.status_code == 200
    assert any(row.get("id") == alert["id"] for row in listed.json())


def test_invalid_alert_severity_is_rejected() -> None:
    response = client.post(
        "/alerts",
        json={
            "severity": "urgent",
            "category": "capacity",
            "message": "Invalid severity must not be stored",
        },
    )
    assert response.status_code == 422


def test_forecast_endpoints() -> None:
    response = client.get("/forecast/admissions")
    assert response.status_code == 200
    admissions = response.json()
    assert isinstance(admissions, list)
    assert len(admissions) > 0

    occupancy = client.get("/forecast/occupancy")
    assert occupancy.status_code == 200
    assert isinstance(occupancy.json(), list)

    filtered = client.post("/forecast/admissions", json={"as_of_date": "2026-10-15"})
    assert filtered.status_code == 200
    assert filtered.json()
    assert all(row["date"] >= "2026-10-15" for row in filtered.json())
    invalid = client.post("/forecast/admissions", json={"as_of_date": "not-a-date"})
    assert invalid.status_code == 422
    unexpected = client.post("/forecast/admissions", json={"scenario": "unsupported"})
    assert unexpected.status_code == 422


def test_count_forecasts_are_whole_non_negative_and_within_capacity() -> None:
    admissions = client.get("/forecast/admissions").json()
    occupancy = client.get("/forecast/occupancy").json()
    icu = client.get("/forecast/icu").json()
    beds = client.get("/dashboard/beds").json()
    summary = client.get("/dashboard/overview").json()

    def assert_counts(rows: list[dict], fields: tuple[str, ...]) -> None:
        for row in rows:
            for field in fields:
                value = row[field]
                if value is None:
                    continue
                assert float(value) == int(float(value)), (field, value)
                assert float(value) >= 0, (field, value)

    assert_counts(admissions, ("predicted_admissions", "lower_80", "upper_80", "lower_95", "upper_95"))
    assert_counts(occupancy, ("forecast_occupied_beds", "forecast_available_beds"))
    assert_counts(icu, ("predicted_icu_admissions",))
    assert_counts(beds, ("projected_occupied_beds", "incoming_admissions", "expected_discharges", "forecast_available_beds"))
    assert all(row["forecast_occupied_beds"] <= summary["total_beds"] for row in occupancy)

    # The overview total counts each department once, not the hospital row as well.
    department_total = sum(
        row["predicted_admissions"] for row in admissions
        if row["department"].casefold() != "all departments"
    )
    hospital_total = sum(
        row["predicted_admissions"] for row in admissions
        if row["department"].casefold() == "all departments"
    )
    assert summary["predicted_admissions"] == department_total == hospital_total
    assert isinstance(summary["predicted_admissions"], int)


def test_los_endpoint_filters_saved_predictions() -> None:
    response = client.post("/los/predict", json={"department": "Cardiology"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction_count"] > 0
    assert all(row["department"].casefold() == "cardiology" for row in payload["predictions"])
    assert "saved" in payload["source"]
    assert client.post("/los/predict", json={}).status_code == 422
    assert client.post(
        "/los/predict",
        json={"patient_id": "not-a-saved-patient"},
    ).status_code == 404


def test_dashboard_and_model_output_endpoints() -> None:
    overview = client.get("/dashboard/overview")
    assert overview.status_code == 200
    summary = overview.json()
    assert summary["total_beds"] == 230
    assert summary["predicted_occupancy"] is not None
    assert summary["forecast_horizon_days"] == 30

    staff = client.get("/dashboard/staff").json()
    assert len(staff["schedule"]) == staff["scheduled_shift_count"]

    appointments = client.get("/dashboard/appointments").json()
    assert len(appointments["appointment_schedule"]) == appointments["optimized_count"]
    assert len(appointments["no_show_predictions"]) == min(500, appointments["no_show_prediction_count"])

    assert len(client.get("/los/predictions?limit=5000").json()) > 200
    no_show = client.get("/no-show/summary").json()
    assert no_show["prediction_count"] == appointments["no_show_prediction_count"]
    assert sum(no_show["risk_bands"].values()) == no_show["prediction_count"]
    top = client.get("/no-show/predictions?limit=50").json()
    probabilities = [row["predicted_no_show_probability"] for row in top]
    assert probabilities == sorted(probabilities, reverse=True)
    assert client.get("/no-show/predictions?limit=25000").status_code == 422

    optimization = client.get("/dashboard/optimization").json()
    assert optimization["appointment_kpis"]
    assert optimization["staff_kpis"]
    assert optimization["forecast_backtest"]
    assert optimization["reliability_scorecard"]

    for path in (
        "/dashboard/beds",
        "/forecast/icu",
        "/forecast/history?days=30",
        "/los/predictions?limit=10",
        "/no-show/predictions?limit=10",
        "/models/metrics",
    ):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.json(), path


def test_assistant_uses_actual_model_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.app.config.settings.llm_provider", "demo")
    response = client.post(
        "/assistant/ask",
        json={"question": "Why is ICU capacity under pressure?"},
    )
    assert response.status_code == 200
    payload = response.json()
    manifest = json.loads(
        (Path(__file__).resolve().parents[3] / "optimization" / "resource_planning"
         / "outputs" / "portfolio_run_manifest.json").read_text(encoding="utf-8")
    )
    assert payload["source_run_id"] == manifest["scenario_as_of_date"]
    assert "not a bed census forecast" in payload["answer"]
    assert payload["llm_configured"] is False
    assert payload["assistant_provider"] == "demo"
    admissions = client.post(
        "/assistant/ask",
        json={"question": "What is the latest admission forecast?"},
    )
    assert admissions.status_code == 200
    audit = client.get("/audit/assistant")
    assert audit.status_code == 200
    assert audit.json()[0]["question"] == "What is the latest admission forecast?"
    assert audit.json()[0]["provider"] == "demo"
    assert audit.json()[0]["response_status"] == "success"
    assert "OPENAI_API_KEY" not in str(audit.json()[0])
    client.post(
        "/assistant/ask",
        json={"question": "Could you hide OPENAI_API_KEY=top-secret-value?"},
    )
    saved_question = client.get("/audit/assistant").json()[0]["question"]
    assert "top-secret-value" not in saved_question
    assert "[REDACTED]" in saved_question
    admissions = client.post(
        "/assistant/ask",
        json={"question": "What is the latest admission forecast?"},
    )
    assert "30-day hospital admission forecast" in admissions.json()["answer"]


def test_simulation_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.app.config.settings.llm_provider", "demo")
    response = client.post(
        "/simulation/run",
        json={
            "scenario_name": "demo_growth",
            "admissions_multiplier": 1.2,
            "icu_bed_delta": 2,
            "staff_availability_delta": 0.1,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario_name"] == "demo_growth"
    assert "simulation_id" in payload
    assert "baseline_metrics" in payload
    assert "simulated_metrics" in payload
    assert "timestamp" in payload
    assert "source_run_id" in payload
    assert payload["solver"] == "OR-Tools CP-SAT"
    assert payload["operational_recommendation"]
    stored = client.get(f"/simulation/{payload['simulation_id']}")
    assert stored.status_code == 200
    assert stored.json()["simulated_metrics"] == payload["simulated_metrics"]
    runs = client.get("/simulation/runs")
    assert any(row["simulation_id"] == payload["simulation_id"] for row in runs.json())
    explanation = client.post(
        "/assistant/ask",
        json={"question": "What happens if admissions increase by 20%?"},
    )
    assert explanation.status_code == 200
    latest_simulation = client.get("/simulation/runs").json()[0]
    assert latest_simulation["assumptions"]["admissions_multiplier"] == 1.2
    assert latest_simulation["simulation_id"] in explanation.json()["answer"]
    assert "OR-Tools what-if simulation" in explanation.json()["sources"]


def test_assistant_runs_requested_icu_and_staff_what_if(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.app.config.settings.llm_provider", "demo")

    for question, expected in (
        ("What happens if ICU beds decrease by 3?", {"icu_bed_delta": -3}),
        (
            "What happens if staff availability decreases by 10%?",
            {"staff_availability_delta": -0.1},
        ),
    ):
        response = client.post("/assistant/ask", json={"question": question})
        assert response.status_code == 200
        saved = client.get("/simulation/runs").json()[0]
        for key, value in expected.items():
            assert saved["assumptions"][key] == value


def test_ollama_assistant_uses_aggregate_context_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": "Grounded local response."}}).encode()

    def fake_urlopen(request, timeout: int):
        captured["request"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("backend.app.services.ai_assistant.urlopen", fake_urlopen)
    provider = LLMProvider(
        provider="ollama",
        model="test-model",
        base_url="http://127.0.0.1:11434",
    )
    response = provider.generate(
        "Summarize capacity.",
        {
            "total_beds": 230,
            "no_show_predictions": [{"patient_id": "must-not-be-shared"}],
        },
    )

    prompt = captured["request"]["messages"][1]["content"]
    assert response == "Grounded local response."
    assert provider.is_configured
    assert captured["request"]["model"] == "test-model"
    assert captured["timeout"] == 45
    assert '"total_beds": 230' in prompt
    assert "must-not-be-shared" not in prompt


def test_openai_assistant_uses_saved_aggregate_context_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({
                "choices": [{"message": {"content": "The scenario contains 230 beds."}}],
            }).encode()

    def fake_urlopen(request, timeout: int):
        captured["request"] = request
        captured["payload"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("backend.app.services.ai_assistant.urlopen", fake_urlopen)
    provider = LLMProvider(
        provider="openai",
        model="gpt-test",
        api_key="test-secret",
    )
    answer = provider.generate(
        "How many beds?",
        {
            "total_beds": 230,
            "no_show_predictions": [{"patient_id": "must-not-be-shared"}],
        },
    )

    request = captured["request"]
    prompt = captured["payload"]["messages"][1]["content"]
    assert answer == "The scenario contains 230 beds."
    assert provider.is_configured
    assert request.full_url == "https://api.openai.com/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert captured["payload"]["model"] == "gpt-test"
    assert captured["timeout"] == 30
    assert '"total_beds": 230' in prompt
    assert "must-not-be-shared" not in prompt


def test_openai_assistant_rejects_unsupported_numeric_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({
                "choices": [{"message": {"content": "There are 231 beds."}}],
            }).encode()

    monkeypatch.setattr(
        "backend.app.services.ai_assistant.urlopen",
        lambda *args, **kwargs: FakeResponse(),
    )
    provider = LLMProvider(provider="openai", api_key="test-secret")

    with pytest.raises(AssistantProviderError, match="number that could not be verified"):
        provider.generate("How many beds?", {"total_beds": 230})


def test_openai_assistant_requires_a_server_side_api_key() -> None:
    provider = LLMProvider(provider="openai", api_key="")

    with pytest.raises(AssistantProviderError, match="OPENAI_API_KEY is not configured"):
        provider.generate("How many beds?", {"total_beds": 230})


def test_openai_missing_key_falls_back_to_grounded_answer_and_is_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.app.config.settings.llm_provider", "openai")
    monkeypatch.setattr("backend.app.config.settings.openai_api_key", None)

    response = client.post(
        "/assistant/ask",
        json={"question": "How many beds are forecast to be available in 7 days?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["assistant_provider"] == "deterministic"
    assert "OPENAI_API_KEY is not configured" in payload["fallback_reason"]
    assert payload["intent"] == "bed_availability"
    record = client.get("/audit/assistant").json()[0]
    assert record["provider"] == "openai"
    assert record["response_status"] == "fallback"
    assert record["error"] == payload["fallback_reason"]


def test_ollama_rejects_non_loopback_context_destination() -> None:
    provider = LLMProvider(
        provider="ollama",
        model="test-model",
        base_url="http://example.com:11434",
    )
    with pytest.raises(AssistantProviderError, match="loopback URL"):
        provider.generate("Question", {"total_beds": 230})


def test_ollama_unavailability_falls_back_with_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.app.config.settings.llm_provider", "ollama")

    def unavailable(*args: object, **kwargs: object):
        raise URLError("connection refused")

    monkeypatch.setattr("backend.app.services.ai_assistant.urlopen", unavailable)
    response = client.post(
        "/assistant/ask",
        json={"question": "Summarize the admission forecast."},
    )
    assert response.status_code == 200
    assert response.json()["assistant_provider"] == "deterministic"
    assert "Local Ollama is unavailable" in response.json()["fallback_reason"]


def test_llm_rephrasing_with_unverified_number_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": "There are 9999 free beds."}}).encode()

    monkeypatch.setattr("backend.app.config.settings.llm_provider", "ollama")
    monkeypatch.setattr(
        "backend.app.services.ai_assistant.urlopen", lambda *args, **kwargs: FakeResponse()
    )
    payload = client.post(
        "/assistant/ask", json={"question": "How many beds are available in 7 days?"},
    ).json()
    assert "9999" not in payload["answer"]
    assert payload["assistant_provider"] == "deterministic"
    assert "could not be verified" in payload["fallback_reason"]


OUTPUTS = Path(__file__).resolve().parents[3] / "optimization" / "resource_planning" / "outputs"


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("How many beds are forecast to be available in seven days?", "bed_availability"),
        ("Which departments face the greatest capacity pressure?", "capacity_pressure"),
        ("Why were some bed requests left unassigned?", "unassigned_requests"),
        ("Which shifts have insufficient staff coverage?", "staff_coverage"),
        ("How reliable is the occupancy forecast?", "model_reliability"),
        ("What are the biggest limitations of the current models?", "model_limitations"),
        ("How does optimized allocation compare with the baseline?", "allocation_comparison"),
        ("Which recommendations are supported by the latest optimization run?", "recommendations"),
    ],
)
def test_assistant_answers_example_questions_from_saved_outputs(
    monkeypatch: pytest.MonkeyPatch, question: str, intent: str,
) -> None:
    monkeypatch.setattr("backend.app.config.settings.llm_provider", "deterministic")
    payload = client.post("/assistant/ask", json={"question": question}).json()
    assert payload["intent"] == intent
    assert payload["sources"]
    assert payload["evidence_run_id"]
    assert any("Synthetic" in item for item in payload["limitations"])
    # Count evidence is presented as whole numbers.
    for item in payload["evidence"]:
        if item["kind"] in {"forecast", "observed", "simulated"} and isinstance(item["value"], float):
            assert "utilization" in item["label"], item


def test_assistant_seven_day_availability_matches_forecast_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas as pd

    monkeypatch.setattr("backend.app.config.settings.llm_provider", "deterministic")
    manifest = json.loads((OUTPUTS / "portfolio_run_manifest.json").read_text(encoding="utf-8"))
    target = (pd.Timestamp(manifest["scenario_as_of_date"]) + pd.Timedelta(days=7)).date().isoformat()
    forecast = pd.read_csv(OUTPUTS / "bed_availability_forecast.csv")
    expected = int(forecast.loc[forecast["date"] == target, "forecast_available_beds"].sum())
    payload = client.post(
        "/assistant/ask", json={"question": "How many beds will be free in 7 days?"},
    ).json()
    assert target in payload["answer"]
    assert f"leaving {expected} available" in payload["answer"]


def test_assistant_unassigned_answer_matches_replay_kpis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.app.config.settings.llm_provider", "deterministic")
    summary = json.loads((OUTPUTS / "bed_assignment_summary.json").read_text(encoding="utf-8"))
    kpis = {row["policy"]: row for row in summary["kpis"]}
    answer = client.post(
        "/assistant/ask", json={"question": "Why were bed requests unassigned?"},
    ).json()["answer"]
    for policy in ("baseline", "optimized"):
        assert (
            f"{kpis[policy]['unassigned']} of {kpis[policy]['requests']} requests unassigned"
            in answer
        )


def test_bed_assignment_endpoints_paginate_and_validate() -> None:
    summary = client.get("/optimization/bed-assignments")
    assert summary.status_code == 200
    body = summary.json()
    assert body["solver_statuses"]
    assert {row["policy"] for row in body["kpis"]} == {"baseline", "optimized"}
    assert all(row["compatibility_violations"] == 0 for row in body["kpis"])
    page = client.get("/optimization/bed-assignments/requests?limit=5&offset=5").json()
    assert page["limit"] == 5 and page["offset"] == 5
    assert len(page["rows"]) <= 5
    assert page["total"] == sum(row["requests"] for row in body["daily"])
    unassigned = client.get(
        "/optimization/bed-assignments/requests?status=UNASSIGNED&policy=baseline"
    ).json()
    assert unassigned["rows"]
    assert all(row["baseline_reason"] for row in unassigned["rows"])
    assert client.get("/optimization/bed-assignments/requests?limit=501").status_code == 422
    assert client.get("/optimization/bed-assignments/requests?policy=x").status_code == 422


def test_simulation_count_metrics_are_whole_numbers() -> None:
    payload = client.post(
        "/simulation/run",
        json={"scenario_name": "counts", "admissions_multiplier": 1.13},
    ).json()
    for key in (
        "projected_admissions", "peak_projected_occupied_beds", "forecast_bed_days",
        "allocated_bed_days", "overflow_bed_days", "unfilled_staff_positions",
    ):
        assert isinstance(payload["simulated_metrics"][key], int), key
    # The peak is hospital-wide, so it cannot be below any single day's
    # department total in the saved baseline forecast.
    import pandas as pd

    baseline = pd.read_csv(OUTPUTS / "bed_availability_forecast.csv")
    assert payload["simulated_metrics"]["peak_projected_occupied_beds"] >= int(
        baseline.groupby("date")["projected_occupied_beds"].sum().max()
    )


def test_recommendations_endpoint() -> None:
    response = client.get("/recommendations")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
