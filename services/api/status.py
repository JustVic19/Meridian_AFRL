"""Database and pgvector checks for GET /system/status."""

from typing import Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from api import __version__
from schemas.system import SystemStatusResponse


def check_system(engine: Engine) -> SystemStatusResponse:
    database: Literal["ok", "unavailable"] = "unavailable"
    pgvector: Literal["available", "unavailable"] = "unavailable"
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            database = "ok"
            present = conn.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            ).scalar()
            if present is True:
                pgvector = "available"
    except SQLAlchemyError:
        database = "unavailable"
        pgvector = "unavailable"
    status: Literal["ok", "degraded"] = (
        "ok" if database == "ok" and pgvector == "available" else "degraded"
    )
    return SystemStatusResponse(
        status=status,
        api="ok",
        database=database,
        pgvector=pgvector,
        version=__version__,
    )
