"""Researcher agent contracts. Findings stay structured and carry model metadata."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ResearchModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ResearchRequest(ResearchModel):
    question: str
    decision_timestamp: datetime
    data_cutoff: datetime

    @field_validator("question")
    @classmethod
    def question_is_present(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question is required")
        return cleaned

    @field_validator("decision_timestamp", "data_cutoff")
    @classmethod
    def timestamp_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value


class KnownFact(ResearchModel):
    statement: str = Field(min_length=1)
    evidence_ids: list[UUID] = Field(min_length=1)
    kind: Literal["observed", "historical_analogue"]


class CandidateHypothesis(ResearchModel):
    statement: str = Field(min_length=1)
    role: Literal["primary", "alternative"]


class ResearchFindings(ResearchModel):
    research_summary: str = Field(min_length=1)
    known_facts: list[KnownFact]
    unknowns: list[str]
    mechanisms: list[str] = Field(min_length=1)
    potential_variables: list[str] = Field(min_length=1)
    candidate_hypotheses: list[CandidateHypothesis] = Field(min_length=1)
    falsification_conditions: list[str] = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)

    @field_validator(
        "unknowns",
        "mechanisms",
        "potential_variables",
        "falsification_conditions",
    )
    @classmethod
    def statements_are_present(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("statement is required")
        return values

    @model_validator(mode="after")
    def findings_are_complete(self) -> Self:
        roles = {item.role for item in self.candidate_hypotheses}
        if "primary" not in roles or "alternative" not in roles:
            raise ValueError("findings need a primary and an alternative hypothesis")
        return self
