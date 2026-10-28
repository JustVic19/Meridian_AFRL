"""Research-lab contracts. These mirror the V1 PostgreSQL tables."""

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

JsonDocument = dict[str, Any] | list[Any]


class HypothesisStatus(StrEnum):
    DRAFT = "DRAFT"
    UNDER_TEST = "UNDER_TEST"
    FAILED = "FAILED"
    VALIDATING = "VALIDATING"
    PAPER_TRADING = "PAPER_TRADING"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


class ArtifactType(StrEnum):
    RESEARCH_NOTE = "RESEARCH_NOTE"
    HYPOTHESIS = "HYPOTHESIS"
    EXPERIMENT_SPEC = "EXPERIMENT_SPEC"
    CODE = "CODE"
    BACKTEST_REPORT = "BACKTEST_REPORT"
    SKEPTIC_REPORT = "SKEPTIC_REPORT"
    VALIDATION_REPORT = "VALIDATION_REPORT"
    FINAL_REPORT = "FINAL_REPORT"
    POSTMORTEM = "POSTMORTEM"


HYPOTHESIS_STATUSES: tuple[str, ...] = tuple(status.value for status in HypothesisStatus)
ARTIFACT_TYPES: tuple[str, ...] = tuple(artifact.value for artifact in ArtifactType)


def sql_in(column: str, values: tuple[str, ...]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"{column} IN ({quoted})"


def sql_date_order(start: str, end: str) -> str:
    return f"{start} IS NULL OR {end} IS NULL OR {start} <= {end}"


class ResearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid", from_attributes=True)


class CompanyRecord(ResearchModel):
    id: UUID
    ticker: str
    cik: str | None = None
    name: str
    exchange: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: Decimal | None = None
    created_at: datetime | None = None


class DocumentRecord(ResearchModel):
    id: UUID
    company_id: UUID | None = None
    document_type: str
    accession_number: str | None = None
    filing_date: date | None = None
    period_start: date | None = None
    period_end: date | None = None
    published_at: datetime | None = None
    effective_at: datetime | None = None
    retrieved_at: datetime | None = None
    source_url: str | None = None
    content: str | None = None
    content_hash: str | None = None
    created_at: datetime | None = None


class ObservationRecord(ResearchModel):
    id: UUID
    company_id: UUID | None = None
    event_type: str
    title: str
    description: str | None = None
    evidence: JsonDocument | None = None
    confidence: Decimal | None = None
    detected_at: datetime | None = None
    published_at: datetime | None = None
    effective_at: datetime | None = None
    period_start: date | None = None
    period_end: date | None = None
    retrieved_at: datetime | None = None
    created_at: datetime | None = None


class HypothesisRecord(ResearchModel):
    id: UUID
    title: str
    statement: str
    universe: JsonDocument | None = None
    mechanism: str | None = None
    entry_rule: JsonDocument | None = None
    exit_rule: JsonDocument | None = None
    expected_horizon: int | None = None
    parent_hypothesis_id: UUID | None = None
    status: HypothesisStatus
    created_at: datetime | None = None

    @model_validator(mode="after")
    def parent_is_not_self(self) -> "HypothesisRecord":
        if self.parent_hypothesis_id == self.id:
            raise ValueError("parent_hypothesis_id must differ from id")
        return self


class ExperimentRecord(ResearchModel):
    id: UUID
    hypothesis_id: UUID
    dataset_version: str
    data_cutoff: datetime | None = None
    code_version: str
    experiment_timestamp: datetime | None = None
    train_start: date | None = None
    train_end: date | None = None
    validation_start: date | None = None
    validation_end: date | None = None
    test_start: date | None = None
    test_end: date | None = None
    specification: JsonDocument | None = None
    parameters: JsonDocument | None = None
    metrics: JsonDocument | None = None
    statistical_tests: JsonDocument | None = None
    status: str
    parent_experiment_id: UUID | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime | None = None

    @model_validator(mode="after")
    def windows_and_parent_are_ordered(self) -> "ExperimentRecord":
        if self.parent_experiment_id == self.id:
            raise ValueError("parent_experiment_id must differ from id")
        pairs = (
            (self.train_start, self.train_end, "train"),
            (self.validation_start, self.validation_end, "validation"),
            (self.test_start, self.test_end, "test"),
        )
        for start, end, label in pairs:
            if start is not None and end is not None and start > end:
                raise ValueError(f"{label} start must be on or before {label} end")
        return self


class ResearchArtifactRecord(ResearchModel):
    id: UUID
    experiment_id: UUID | None = None
    artifact_type: ArtifactType
    content: str
    model_name: str | None = None
    model_version: str | None = None
    prompt_version: str | None = None
    created_at: datetime | None = None


class StrategyRecord(ResearchModel):
    id: UUID
    experiment_id: UUID | None = None
    name: str
    version: int = Field(default=1, ge=1)
    signal_definition: JsonDocument | None = None
    portfolio_rules: JsonDocument | None = None
    risk_rules: JsonDocument | None = None
    capacity_estimate: Decimal | None = None
    status: str | None = None
    created_at: datetime | None = None
