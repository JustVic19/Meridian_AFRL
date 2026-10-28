"""FastAPI application. Liveness is independent of the database."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from api import __version__
from api.db import get_engine
from api.status import check_system
from schemas.system import HealthResponse, SystemStatusResponse

app = FastAPI(title="Financial Intelligence Lab API")


def build_status() -> SystemStatusResponse:
    try:
        return check_system(get_engine())
    except (SQLAlchemyError, RuntimeError, OSError):
        return SystemStatusResponse(
            status="degraded",
            api="ok",
            database="unavailable",
            pgvector="unavailable",
            version=__version__,
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/system/status")
def system_status() -> JSONResponse:
    result = build_status()
    code = 200 if result.status == "ok" else 503
    return JSONResponse(status_code=code, content=result.model_dump())
