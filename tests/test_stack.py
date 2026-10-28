"""Live stack checks. Requires `docker compose up` and does not skip."""

import os

import httpx
from sqlalchemy import create_engine, text

HOST_DATABASE_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/afil"


def test_api_starts() -> None:
    response = httpx.get("http://localhost:8000/health", timeout=5.0)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_system_status_works() -> None:
    response = httpx.get("http://localhost:8000/system/status", timeout=5.0)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["api"] == "ok"
    assert body["database"] == "ok"
    assert body["pgvector"] == "available"


def test_frontend_starts() -> None:
    response = httpx.get("http://localhost:3000", timeout=10.0)
    assert response.status_code == 200


def test_postgres_reachable_and_pgvector_available() -> None:
    url = os.getenv("AFIL_TEST_DATABASE_URL", HOST_DATABASE_URL)
    engine = create_engine(url)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        present = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        ).scalar()
    assert present is True
