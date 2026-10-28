"""Persist filings once. A repeated accession returns the original row."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol
from uuid import UUID, uuid5

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models import Company, Document
from data.sec.normalize import NormalizedFiling

_NAMESPACE = UUID("a1f1c0de-5ec0-4000-8000-000000000001")


def stable_id(kind: str, key: str) -> UUID:
    """Same accession and CIK always map to the same ids."""
    return uuid5(_NAMESPACE, f"{kind}:{key}")


@dataclass(frozen=True)
class StoredFiling:
    id: UUID
    company_id: UUID
    ticker: str
    cik: str
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


class FilingStore(Protocol):
    def get(self, accession_number: str) -> StoredFiling | None:
        """Return the filing already stored for this accession."""

    def save(self, filing: NormalizedFiling) -> StoredFiling:
        """Insert one filing. Callers skip accessions that already exist."""


class MemoryFilingStore:
    def __init__(self) -> None:
        self._rows: dict[str, StoredFiling] = {}

    @property
    def filings(self) -> tuple[StoredFiling, ...]:
        return tuple(self._rows.values())

    def get(self, accession_number: str) -> StoredFiling | None:
        return self._rows.get(accession_number)

    def save(self, filing: NormalizedFiling) -> StoredFiling:
        stored = _from_normalized(filing)
        self._rows[filing.accession_number] = stored
        return stored


class SqlFilingStore:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, accession_number: str) -> StoredFiling | None:
        document = self._session.scalar(
            select(Document).where(Document.accession_number == accession_number)
        )
        if document is None or document.company_id is None:
            return None
        company = self._session.get(Company, document.company_id)
        if company is None or company.cik is None:
            return None
        return _from_rows(document, company)

    def save(self, filing: NormalizedFiling) -> StoredFiling:
        company = self._session.scalar(select(Company).where(Company.cik == filing.cik))
        if company is None:
            company = Company(
                id=stable_id("company", filing.cik),
                ticker=filing.ticker,
                cik=filing.cik,
                name=filing.company_name,
                exchange=filing.exchange,
            )
            self._session.add(company)
            self._session.flush()
        document = Document(
            id=stable_id("document", filing.accession_number),
            company_id=company.id,
            document_type=filing.document_type,
            accession_number=filing.accession_number,
            filing_date=filing.filing_date,
            period_start=filing.period_start,
            period_end=filing.period_end,
            published_at=filing.published_at,
            effective_at=filing.effective_at,
            retrieved_at=filing.retrieved_at,
            source_url=filing.source_url,
            content=filing.content,
            content_hash=filing.content_hash,
        )
        self._session.add(document)
        self._session.flush()
        return _from_normalized(filing)


def _from_normalized(filing: NormalizedFiling) -> StoredFiling:
    return StoredFiling(
        id=stable_id("document", filing.accession_number),
        company_id=stable_id("company", filing.cik),
        ticker=filing.ticker,
        cik=filing.cik,
        document_type=filing.document_type,
        accession_number=filing.accession_number,
        filing_date=filing.filing_date,
        period_start=filing.period_start,
        period_end=filing.period_end,
        published_at=filing.published_at,
        effective_at=filing.effective_at,
        retrieved_at=filing.retrieved_at,
        source_url=filing.source_url,
        content=filing.content,
        content_hash=filing.content_hash,
    )


def _from_rows(document: Document, company: Company) -> StoredFiling:
    if (
        document.filing_date is None
        or document.period_start is None
        or document.period_end is None
        or document.published_at is None
        or document.effective_at is None
        or document.retrieved_at is None
        or document.source_url is None
        or document.content is None
        or document.content_hash is None
        or document.accession_number is None
        or company.cik is None
    ):
        raise ValueError("stored filing is missing temporal metadata")
    return StoredFiling(
        id=document.id,
        company_id=company.id,
        ticker=company.ticker,
        cik=company.cik,
        document_type=document.document_type,
        accession_number=document.accession_number,
        filing_date=document.filing_date,
        period_start=document.period_start,
        period_end=document.period_end,
        published_at=document.published_at,
        effective_at=document.effective_at,
        retrieved_at=document.retrieved_at,
        source_url=document.source_url,
        content=document.content,
        content_hash=document.content_hash,
    )
