from __future__ import annotations

import json
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
    assert len(appointments["no_show_predictions"]) == appointments["no_show_prediction_count"]

    assert len(client.get("/los/predictions?limit=5000").json()) > 200
    assert len(client.get("/no-show/predictions?limit=25000").json()) == appointments["no_show_prediction_count"]

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
    assert payload["source_run_id"] == "2026-10-04"
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
    assert "30-day admission forecast" in admissions.json()["answer"]


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


def test_openai_missing_key_is_reported_and_audited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.app.config.settings.llm_provider", "openai")
    monkeypatch.setattr("backend.app.config.settings.openai_api_key", None)

    response = client.post(
        "/assistant/ask",
        json={"question": "How many beds?"},
    )

    assert response.status_code == 503
    assert "OPENAI_API_KEY is not configured" in response.json()["detail"]
    record = client.get("/audit/assistant").json()[0]
    assert record["provider"] == "openai"
    assert record["response_status"] == "error"
    assert record["error"] == response.json()["detail"]


def test_ollama_rejects_non_loopback_context_destination() -> None:
    provider = LLMProvider(
        provider="ollama",
        model="test-model",
        base_url="http://example.com:11434",
    )
    with pytest.raises(AssistantProviderError, match="loopback URL"):
        provider.generate("Question", {"total_beds": 230})


def test_ollama_unavailability_is_reported_without_demo_fallback(
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
    assert response.status_code == 503
    assert "Local Ollama is unavailable" in response.json()["detail"]


def test_recommendations_endpoint() -> None:
    response = client.get("/recommendations")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
