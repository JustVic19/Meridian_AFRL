"""Temporal metadata for a filing. Availability is the SEC acceptance time."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time

from data.sec.parse import FilingRef, MalformedFiling


@dataclass(frozen=True)
class TemporalMetadata:
    period_start: date
    period_end: date
    published_at: datetime
    effective_at: datetime
    retrieved_at: datetime


def temporal_metadata(
    filing: FilingRef,
    period_start: date,
    retrieved_at: datetime,
) -> TemporalMetadata:
    """Preserve filing, period, publication, and retrieval times without reading the clock."""
    if retrieved_at.tzinfo is None or retrieved_at.utcoffset() is None:
        raise MalformedFiling("retrieved_at")
    return TemporalMetadata(
        period_start=period_start,
        period_end=filing.report_date,
        published_at=filing.published_at,
        effective_at=datetime.combine(filing.filing_date, time.min, tzinfo=UTC),
        retrieved_at=retrieved_at,
    )
