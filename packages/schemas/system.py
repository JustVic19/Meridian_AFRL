"""System endpoint contracts. No financial fields."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]


class SystemStatusResponse(BaseModel):
    status: Literal["ok", "degraded"]
    api: Literal["ok"]
    database: Literal["ok", "unavailable"]
    pgvector: Literal["available", "unavailable"]
    version: str
