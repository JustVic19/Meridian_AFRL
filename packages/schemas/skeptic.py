"""Skeptic contracts. The verdict is a grade over recorded evidence."""

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SkepticModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckName(StrEnum):
    LOOK_AHEAD_BIAS = "look_ahead_bias"
    SURVIVORSHIP_BIAS = "survivorship_bias"
    SELECTION_BIAS = "selection_bias"
    DATA_LEAKAGE = "data_leakage"
    MULTIPLE_TESTING = "multiple_testing"
    OVERFITTING = "overfitting"
    SAMPLE_SIZE = "sample_size"
    REGIME_DEPENDENCE = "regime_dependence"
    TRANSACTION_COSTS = "transaction_costs"
    SLIPPAGE = "slippage"
    LIQUIDITY = "liquidity"
    CAPACITY = "capacity"
    BENCHMARK_SELECTION = "benchmark_selection"
    ECONOMIC_PLAUSIBILITY = "economic_plausibility"


class CheckGrade(StrEnum):
    PASS = "PASS"
    CONDITIONAL = "CONDITIONAL"
    FAIL = "FAIL"


class Verdict(StrEnum):
    PASS = "PASS"
    CONDITIONAL_PASS = "CONDITIONAL_PASS"
    FAIL = "FAIL"


class UniverseMember(SkepticModel):
    symbol: str = Field(min_length=1)
    listed_on: date
    delisted_on: date | None = None


class RegimeWindow(SkepticModel):
    name: str = Field(min_length=1)
    start: date
    end: date

    @model_validator(mode="after")
    def end_follows_start(self) -> Self:
        if self.end < self.start:
            raise ValueError("regime end is before regime start")
        return self


class AuditFacts(SkepticModel):
    """Facts the skeptic may cite. The model cannot add to this record."""

    members: list[UniverseMember]
    retained_delisted: bool
    eligible_symbols: list[str]
    selected_symbols: list[str]
    selection_rule: str = ""
    feature_timestamps: list[datetime]
    decision_timestamp: datetime
    normalization_window: Literal["train", "full_sample", "test"]
    comparisons_tested: int = Field(ge=1)
    multiple_testing_adjustment: Literal["none", "bonferroni", "holm", "bh"]
    specifications_searched: int = Field(ge=1)
    regimes: list[RegimeWindow]
    average_dollar_volume: dict[str, Decimal]
    order_notional: Decimal = Field(ge=0)
    participation_limit: Decimal = Field(gt=0, le=1)
    capacity_notional: Decimal | None = None
    benchmark_candidates: list[str]


class ExperimentSnapshot(SkepticModel):
    """The stored experiment fields a methodological check is allowed to read."""

    data_cutoff: datetime | None
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    specification: dict[str, Any]
    parameters: dict[str, Any]
    statistical_tests: dict[str, Any]
    hypothesis_statement: str
    hypothesis_mechanism: str | None = None


class EvidenceItem(SkepticModel):
    id: UUID
    check: CheckName
    field: str
    observed: str


class CheckFinding(SkepticModel):
    check: CheckName
    grade: CheckGrade
    reason: str = Field(min_length=1)
    evidence_ids: list[UUID] = Field(min_length=1)
    challenge: str = Field(min_length=1)


class SkepticReport(SkepticModel):
    experiment_id: UUID
    verdict: Verdict
    checks: list[CheckFinding]
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def checks_follow_the_method_list(self) -> Self:
        names = [item.check for item in self.checks]
        if names != list(CheckName):
            raise ValueError("checks must cover every method in order")
        return self


class ModelNote(SkepticModel):
    check: CheckName
    evidence_ids: list[UUID] = Field(min_length=1)
    challenge: str = Field(min_length=1)


class ModelNotes(SkepticModel):
    notes: list[ModelNote] = Field(min_length=1)
