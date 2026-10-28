"""HypothesisScientist contracts. A hypothesis is a new row, never an overwrite."""

import re
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from schemas.researcher import ResearchFindings

_VAGUE_STATEMENT = re.compile(r"guidance cuts are bad\.?", re.IGNORECASE)
_VAGUE_BENCHMARKS = frozenset({"market", "the market", "stocks", "bad", "good"})


class HypothesisModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SupportingEvidence(HypothesisModel):
    document_id: UUID
    excerpt: str = Field(min_length=1)
    accession_number: str | None = None


class HypothesisRequest(HypothesisModel):
    question: str
    findings: ResearchFindings
    evidence: list[SupportingEvidence] = Field(min_length=1)
    parent_hypothesis_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def question_is_present(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question is required")
        return cleaned


class UniverseDefinition(HypothesisModel):
    asset_class: Literal["US_EQUITY"]
    min_market_cap: int = Field(gt=0)


class EventDefinition(HypothesisModel):
    type: str = Field(min_length=1)
    threshold: float

    @field_validator("threshold")
    @classmethod
    def threshold_is_finite(cls, value: float) -> float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("event threshold is required")
        return value


class EntryRule(HypothesisModel):
    timing: str = Field(min_length=1)


class ExitRule(HypothesisModel):
    holding_period_days: int = Field(ge=1)


class HypothesisSpecification(HypothesisModel):
    statement: str
    universe: UniverseDefinition
    event_definition: EventDefinition
    entry_rule: EntryRule
    exit_rule: ExitRule
    benchmark: str
    holding_period: int = Field(ge=1)
    mechanism: str = Field(min_length=20)
    falsification_conditions: list[str] = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)

    @field_validator("statement")
    @classmethod
    def statement_is_falsifiable(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if (
            _VAGUE_STATEMENT.fullmatch(cleaned)
            or not re.search(r"\d", cleaned)
            or len(cleaned.split()) < 8
        ):
            raise ValueError("vague hypothesis")
        return cleaned

    @field_validator("benchmark")
    @classmethod
    def benchmark_is_specific(cls, value: str) -> str:
        cleaned = value.strip()
        if cleaned.lower() in _VAGUE_BENCHMARKS:
            raise ValueError("benchmark is not specific")
        return cleaned

    @field_validator("falsification_conditions")
    @classmethod
    def falsifiers_are_present(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("missing falsification conditions")
        return values

    @model_validator(mode="after")
    def horizon_matches_exit(self) -> Self:
        if self.exit_rule.holding_period_days != self.holding_period:
            raise ValueError("horizon does not match the exit rule")
        return self


class FormalizedHypothesis(HypothesisModel):
    id: UUID
    version: int = Field(ge=1)
    parent_hypothesis_id: UUID | None
    specification: HypothesisSpecification
