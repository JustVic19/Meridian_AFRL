"""Normalize one filing into the document fields the laboratory stores."""

import hashlib
from dataclasses import dataclass
from datetime import date, datetime

from data.sec.parse import FilingRef, MalformedFiling
from data.sec.temporal import temporal_metadata


@dataclass(frozen=True)
class NormalizedFiling:
    ticker: str
    cik: str
    company_name: str
    exchange: str | None
    document_type: str
    accession_number: str
    filing_date: date
    period_start: date
    period_end: date
    published_at: datetime
    effective_at: datetime
    retrieved_at: datetime
    source_url: str
    content: str
    content_hash: str


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def normalize(
    filing: FilingRef,
    content: str,
    period_start: date,
    retrieved_at: datetime,
    source_url: str,
    ticker: str,
) -> NormalizedFiling:
    if not content:
        raise MalformedFiling("empty filing")
    metadata = temporal_metadata(filing, period_start, retrieved_at)
    return NormalizedFiling(
        ticker=ticker.strip().upper(),
        cik=filing.cik,
        company_name=filing.company_name,
        exchange=filing.exchange,
        document_type=filing.form,
        accession_number=filing.accession_number,
        filing_date=filing.filing_date,
        period_start=metadata.period_start,
        period_end=metadata.period_end,
        published_at=metadata.published_at,
        effective_at=metadata.effective_at,
        retrieved_at=metadata.retrieved_at,
        source_url=source_url,
        content=content,
        content_hash=content_hash(content),
    )
