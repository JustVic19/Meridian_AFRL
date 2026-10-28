"""Bounded EDGAR ingestion: tickers, submissions, and the filings those submissions name."""

from dataclasses import dataclass
from datetime import date, datetime

from data.sec.client import (
    TICKERS_URL,
    SecClient,
    archive_url,
    company_facts_url,
    submissions_url,
)
from data.sec.normalize import normalize
from data.sec.parse import (
    FilingRef,
    MalformedFiling,
    parse_company_tickers,
    period_starts,
    select_filings,
)
from data.sec.store import FilingStore, StoredFiling

MAX_TICKERS = 25
MAX_FILINGS_PER_COMPANY = 20
ALLOWED_FORMS = frozenset({"10-K", "10-Q", "8-K"})
_FACT_FORMS = frozenset({"10-K", "10-Q"})


class IngestBoundsError(Exception):
    """The request would download more than the pipeline allows."""


@dataclass(frozen=True)
class IngestRequest:
    tickers: tuple[str, ...]
    forms: tuple[str, ...]
    max_filings_per_company: int
    data_cutoff: datetime
    retrieved_at: datetime


@dataclass(frozen=True)
class IngestResult:
    filings: tuple[StoredFiling, ...]
    inserted_accessions: tuple[str, ...]
    skipped_accessions: tuple[str, ...]


def ingest(request: IngestRequest, client: SecClient, store: FilingStore) -> IngestResult:
    """Ingest a named universe. Filings published after data_cutoff are not downloaded."""
    forms = _validate(request)
    mapping = parse_company_tickers(client.get_json(TICKERS_URL))
    stored: list[StoredFiling] = []
    inserted: list[str] = []
    skipped: list[str] = []
    for ticker in request.tickers:
        canonical = ticker.strip().upper()
        cik = mapping.cik_for(canonical)
        selected = select_filings(
            client.get_json(submissions_url(cik)),
            cik=cik,
            forms=forms,
            limit=request.max_filings_per_company,
            data_cutoff=request.data_cutoff,
        )
        missing = [filing for filing in selected if store.get(filing.accession_number) is None]
        periods = (
            period_starts(client.get_json(company_facts_url(cik)))
            if any(filing.form in _FACT_FORMS for filing in missing)
            else {}
        )
        for filing in selected:
            existing = store.get(filing.accession_number)
            if existing is not None:
                skipped.append(filing.accession_number)
                stored.append(existing)
                continue
            source_url = archive_url(filing.cik, filing.accession_number, filing.primary_document)
            normalized = normalize(
                filing,
                client.get_text(source_url),
                _period_start(filing, periods),
                request.retrieved_at,
                source_url,
                canonical,
            )
            saved = store.save(normalized)
            inserted.append(saved.accession_number)
            stored.append(saved)
    return IngestResult(tuple(stored), tuple(inserted), tuple(skipped))


def _validate(request: IngestRequest) -> frozenset[str]:
    if not request.tickers or len(request.tickers) > MAX_TICKERS:
        raise IngestBoundsError("ticker universe must contain 1 to 25 symbols")
    if len(set(request.tickers)) != len(request.tickers):
        raise IngestBoundsError("ticker universe contains a duplicate")
    forms = frozenset(request.forms)
    if not forms or not forms <= ALLOWED_FORMS:
        raise IngestBoundsError("forms must be 10-K, 10-Q, or 8-K")
    if not 1 <= request.max_filings_per_company <= MAX_FILINGS_PER_COMPANY:
        raise IngestBoundsError("max filings per company must be 1 to 20")
    _aware(request.data_cutoff, "data_cutoff")
    _aware(request.retrieved_at, "retrieved_at")
    return forms


def _aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise IngestBoundsError(f"{label} must be timezone-aware")


def _period_start(filing: FilingRef, periods: dict[tuple[str, str], date]) -> date:
    if filing.form == "8-K":
        return filing.report_date
    key = (filing.accession_number, filing.report_date.isoformat())
    try:
        return periods[key]
    except KeyError as exc:
        raise MalformedFiling(filing.accession_number) from exc
