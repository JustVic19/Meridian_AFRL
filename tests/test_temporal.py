"""Point-in-time eligibility. Fixed UTC instants, no database and no model."""

from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest

from research.validation.temporal import (
    Eligibility,
    EligibilityReason,
    LookaheadError,
    TemporalRecord,
    evaluate,
    filter_eligible,
    require_eligible,
)

DECISION = datetime(2022, 5, 5, 14, 30, tzinfo=UTC)
CUTOFF = datetime(2022, 12, 31, 23, 59, tzinfo=UTC)
PERIOD_START = date(2022, 1, 1)
PERIOD_END = date(2022, 3, 31)
EFFECTIVE = datetime(2022, 5, 4, 20, 0, tzinfo=UTC)
RETRIEVED = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class Filing:
    record_id: str
    published_at: datetime | None
    effective_at: datetime | None
    period_start: date | None
    period_end: date | None
    retrieved_at: datetime | None


def _filing(
    record_id: str,
    published_at: datetime | None,
    *,
    effective_at: datetime | None = EFFECTIVE,
    period_start: date | None = PERIOD_START,
    period_end: date | None = PERIOD_END,
    retrieved_at: datetime | None = RETRIEVED,
) -> Filing:
    return Filing(
        record_id=record_id,
        published_at=published_at,
        effective_at=effective_at,
        period_start=period_start,
        period_end=period_end,
        retrieved_at=retrieved_at,
    )


def _ids(
    filings: list[Filing],
    decision: datetime = DECISION,
    cutoff: datetime | None = CUTOFF,
) -> list[str]:
    return [filing.record_id for filing in filter_eligible(filings, decision, cutoff)]


def test_historical_information_is_allowed() -> None:
    published = datetime(2022, 5, 4, 20, 0, tzinfo=UTC)
    decision = evaluate(
        TemporalRecord(published, EFFECTIVE, PERIOD_START, PERIOD_END, RETRIEVED),
        DECISION,
        CUTOFF,
    )
    assert decision.status is Eligibility.ALLOWED
    assert decision.reason is EligibilityReason.ALLOWED


def test_future_information_is_forbidden() -> None:
    future = _filing("future", datetime(2023, 3, 1, 12, 0, tzinfo=UTC))
    decision = evaluate(
        TemporalRecord(
            future.published_at,
            future.effective_at,
            future.period_start,
            future.period_end,
            future.retrieved_at,
        ),
        DECISION,
        CUTOFF,
    )
    assert decision.status is Eligibility.FORBIDDEN
    assert decision.reason is EligibilityReason.PUBLISHED_AFTER_DECISION
    with pytest.raises(LookaheadError, match="PUBLISHED_AFTER_DECISION"):
        require_eligible([future], DECISION, CUTOFF)


def test_same_day_timestamp_edges() -> None:
    morning = datetime(2022, 5, 5, 9, 0, tzinfo=UTC)
    afternoon = datetime(2022, 5, 5, 16, 0, tzinfo=UTC)
    same_instant = DECISION
    assert (
        evaluate(
            TemporalRecord(morning, EFFECTIVE, PERIOD_START, PERIOD_END, RETRIEVED),
            DECISION,
            CUTOFF,
        ).status
        is Eligibility.ALLOWED
    )
    assert (
        evaluate(
            TemporalRecord(afternoon, EFFECTIVE, PERIOD_START, PERIOD_END, RETRIEVED),
            DECISION,
            CUTOFF,
        ).reason
        is EligibilityReason.PUBLISHED_AFTER_DECISION
    )
    assert (
        evaluate(
            TemporalRecord(same_instant, EFFECTIVE, PERIOD_START, PERIOD_END, RETRIEVED),
            DECISION,
            CUTOFF,
        ).status
        is Eligibility.ALLOWED
    )


def test_missing_timestamps_are_forbidden() -> None:
    complete = TemporalRecord(
        datetime(2022, 5, 4, tzinfo=UTC),
        EFFECTIVE,
        PERIOD_START,
        PERIOD_END,
        RETRIEVED,
    )
    cases = (
        (complete, None, EligibilityReason.MISSING_DATA_CUTOFF),
        (
            TemporalRecord(None, EFFECTIVE, PERIOD_START, PERIOD_END, RETRIEVED),
            CUTOFF,
            EligibilityReason.MISSING_PUBLISHED_AT,
        ),
        (
            TemporalRecord(complete.published_at, None, PERIOD_START, PERIOD_END, RETRIEVED),
            CUTOFF,
            EligibilityReason.MISSING_EFFECTIVE_AT,
        ),
        (
            TemporalRecord(complete.published_at, EFFECTIVE, None, PERIOD_END, RETRIEVED),
            CUTOFF,
            EligibilityReason.MISSING_PERIOD_START,
        ),
        (
            TemporalRecord(complete.published_at, EFFECTIVE, PERIOD_START, None, RETRIEVED),
            CUTOFF,
            EligibilityReason.MISSING_PERIOD_END,
        ),
        (
            TemporalRecord(complete.published_at, EFFECTIVE, PERIOD_START, PERIOD_END, None),
            CUTOFF,
            EligibilityReason.MISSING_RETRIEVED_AT,
        ),
        (
            TemporalRecord(
                datetime(2022, 5, 4, 20, 0),
                EFFECTIVE,
                PERIOD_START,
                PERIOD_END,
                RETRIEVED,
            ),
            CUTOFF,
            EligibilityReason.NAIVE_TIMESTAMP,
        ),
    )
    for record, cutoff, reason in cases:
        decision = evaluate(record, DECISION, cutoff)
        assert decision.status is Eligibility.FORBIDDEN
        assert decision.reason is reason


def test_dataset_cutoff_excludes_later_publication() -> None:
    published = datetime(2022, 6, 1, 12, 0, tzinfo=UTC)
    decision = datetime(2022, 7, 1, 12, 0, tzinfo=UTC)
    cutoff = datetime(2022, 5, 31, 23, 59, tzinfo=UTC)
    record = TemporalRecord(published, EFFECTIVE, PERIOD_START, PERIOD_END, RETRIEVED)
    assert evaluate(record, decision, cutoff).reason is EligibilityReason.PUBLISHED_AFTER_CUTOFF
    late_decision = datetime(2022, 8, 1, tzinfo=UTC)
    early = TemporalRecord(
        datetime(2022, 5, 1, tzinfo=UTC),
        EFFECTIVE,
        PERIOD_START,
        PERIOD_END,
        RETRIEVED,
    )
    assert evaluate(early, late_decision, cutoff).reason is EligibilityReason.DECISION_AFTER_CUTOFF


def test_experiment_result_is_reproducible() -> None:
    filings = [
        _filing("early", datetime(2022, 5, 1, 12, 0, tzinfo=UTC)),
        _filing("late", datetime(2022, 5, 6, 12, 0, tzinfo=UTC)),
        _filing("same", DECISION),
    ]
    dataset_version = "sec-2024-01"
    code_version = "a1b2c3d"
    first = _eligible_ids(filings, dataset_version, code_version, datetime(2024, 3, 1, tzinfo=UTC))
    second = _eligible_ids(filings, dataset_version, code_version, datetime(2025, 1, 1, tzinfo=UTC))
    assert first == second == ["early", "same"]


def _eligible_ids(
    filings: list[Filing],
    dataset_version: str,
    code_version: str,
    experiment_timestamp: datetime,
) -> list[str]:
    """Eligibility ignores when the experiment row was written."""
    del dataset_version, code_version, experiment_timestamp
    return _ids(filings)


def test_point_in_time_filter_keeps_input_order() -> None:
    filings = [
        _filing("after", datetime(2022, 5, 6, 9, 0, tzinfo=UTC)),
        _filing("before", datetime(2022, 5, 4, 9, 0, tzinfo=UTC)),
        _filing("equal", DECISION),
        _filing("same-day-later", datetime(2022, 5, 5, 18, 0, tzinfo=UTC)),
    ]
    assert _ids(filings) == ["before", "equal"]


def test_lookahead_uses_publication_time_not_period_or_retrieval() -> None:
    look_ahead = _filing(
        "filed-later",
        datetime(2022, 8, 1, 12, 0, tzinfo=UTC),
        period_end=date(2022, 3, 31),
    )
    already_public = _filing(
        "already-public",
        datetime(2022, 5, 4, 12, 0, tzinfo=UTC),
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert (
        evaluate(
            TemporalRecord(
                look_ahead.published_at,
                look_ahead.effective_at,
                look_ahead.period_start,
                look_ahead.period_end,
                look_ahead.retrieved_at,
            ),
            DECISION,
            CUTOFF,
        ).reason
        is EligibilityReason.PUBLISHED_AFTER_DECISION
    )
    with pytest.raises(LookaheadError):
        require_eligible([look_ahead, already_public], DECISION, CUTOFF)
    assert _ids([already_public, look_ahead]) == ["already-public"]
