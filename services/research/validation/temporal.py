"""Deterministic point-in-time eligibility. No database, clock, or model calls."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Protocol


class HasTemporal(Protocol):
    published_at: datetime | None
    effective_at: datetime | None
    period_start: date | None
    period_end: date | None
    retrieved_at: datetime | None


class Eligibility(StrEnum):
    ALLOWED = "ALLOWED"
    FORBIDDEN = "FORBIDDEN"


class EligibilityReason(StrEnum):
    ALLOWED = "ALLOWED"
    MISSING_PUBLISHED_AT = "MISSING_PUBLISHED_AT"
    MISSING_EFFECTIVE_AT = "MISSING_EFFECTIVE_AT"
    MISSING_PERIOD_START = "MISSING_PERIOD_START"
    MISSING_PERIOD_END = "MISSING_PERIOD_END"
    MISSING_RETRIEVED_AT = "MISSING_RETRIEVED_AT"
    NAIVE_TIMESTAMP = "NAIVE_TIMESTAMP"
    PUBLISHED_AFTER_DECISION = "PUBLISHED_AFTER_DECISION"
    PUBLISHED_AFTER_CUTOFF = "PUBLISHED_AFTER_CUTOFF"
    DECISION_AFTER_CUTOFF = "DECISION_AFTER_CUTOFF"
    MISSING_DATA_CUTOFF = "MISSING_DATA_CUTOFF"


@dataclass(frozen=True)
class TemporalRecord:
    published_at: datetime | None
    effective_at: datetime | None
    period_start: date | None
    period_end: date | None
    retrieved_at: datetime | None


@dataclass(frozen=True)
class EligibilityDecision:
    status: Eligibility
    reason: EligibilityReason


class LookaheadError(Exception):
    """A dataset includes information that was not yet published."""

    def __init__(self, reason: EligibilityReason) -> None:
        self.reason = reason
        super().__init__(reason.value)


def as_temporal(source: HasTemporal) -> TemporalRecord:
    """Read the five temporal fields from a document, observation, or record."""
    return TemporalRecord(
        published_at=source.published_at,
        effective_at=source.effective_at,
        period_start=source.period_start,
        period_end=source.period_end,
        retrieved_at=source.retrieved_at,
    )


def evaluate(
    record: TemporalRecord,
    decision_timestamp: datetime,
    data_cutoff: datetime | None,
) -> EligibilityDecision:
    """Return whether the record was public at the simulated decision."""
    if _has_naive_timestamp(record, decision_timestamp, data_cutoff):
        return _forbidden(EligibilityReason.NAIVE_TIMESTAMP)
    if data_cutoff is None:
        return _forbidden(EligibilityReason.MISSING_DATA_CUTOFF)
    missing = _missing_field(record)
    if missing is not None:
        return _forbidden(missing)
    assert record.published_at is not None
    if record.published_at > decision_timestamp:
        return _forbidden(EligibilityReason.PUBLISHED_AFTER_DECISION)
    if record.published_at > data_cutoff:
        return _forbidden(EligibilityReason.PUBLISHED_AFTER_CUTOFF)
    if decision_timestamp > data_cutoff:
        return _forbidden(EligibilityReason.DECISION_AFTER_CUTOFF)
    return EligibilityDecision(status=Eligibility.ALLOWED, reason=EligibilityReason.ALLOWED)


def filter_eligible[T: HasTemporal](
    records: Sequence[T],
    decision_timestamp: datetime,
    data_cutoff: datetime | None,
) -> list[T]:
    """Keep eligible records in their original order."""
    return [
        record
        for record in records
        if evaluate(as_temporal(record), decision_timestamp, data_cutoff).status
        is Eligibility.ALLOWED
    ]


def require_eligible(
    records: Sequence[HasTemporal],
    decision_timestamp: datetime,
    data_cutoff: datetime | None,
) -> None:
    """Reject a dataset that contains any future or incomplete information."""
    for record in records:
        decision = evaluate(as_temporal(record), decision_timestamp, data_cutoff)
        if decision.status is Eligibility.FORBIDDEN:
            raise LookaheadError(decision.reason)


def _forbidden(reason: EligibilityReason) -> EligibilityDecision:
    return EligibilityDecision(status=Eligibility.FORBIDDEN, reason=reason)


def _has_naive_timestamp(
    record: TemporalRecord,
    decision_timestamp: datetime,
    data_cutoff: datetime | None,
) -> bool:
    instants: list[datetime] = [decision_timestamp]
    if data_cutoff is not None:
        instants.append(data_cutoff)
    for instant in (record.published_at, record.effective_at, record.retrieved_at):
        if instant is not None:
            instants.append(instant)
    return any(_is_naive(instant) for instant in instants)


def _is_naive(instant: datetime) -> bool:
    return instant.tzinfo is None or instant.utcoffset() is None


def _missing_field(record: TemporalRecord) -> EligibilityReason | None:
    if record.published_at is None:
        return EligibilityReason.MISSING_PUBLISHED_AT
    if record.effective_at is None:
        return EligibilityReason.MISSING_EFFECTIVE_AT
    if record.period_start is None:
        return EligibilityReason.MISSING_PERIOD_START
    if record.period_end is None:
        return EligibilityReason.MISSING_PERIOD_END
    if record.retrieved_at is None:
        return EligibilityReason.MISSING_RETRIEVED_AT
    return None
