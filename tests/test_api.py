"""API contract tests. These do not require Docker."""

import pytest
from fastapi.testclient import TestClient

import api.main as main
from schemas.system import SystemStatusResponse


def test_api_app_imports() -> None:
    assert main.app is not None


def test_health() -> None:
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_system_status_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake() -> SystemStatusResponse:
        return SystemStatusResponse(
            status="ok",
            api="ok",
            database="ok",
            pgvector="available",
            version="0.1.0",
        )

    monkeypatch.setattr(main, "build_status", fake)
    client = TestClient(main.app)
    response = client.get("/system/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "api", "database", "pgvector", "version"}
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["pgvector"] == "available"


def test_system_status_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake() -> SystemStatusResponse:
        return SystemStatusResponse(
            status="degraded",
            api="ok",
            database="unavailable",
            pgvector="unavailable",
            version="0.1.0",
        )

    monkeypatch.setattr(main, "build_status", fake)
    client = TestClient(main.app)
    response = client.get("/system/status")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
