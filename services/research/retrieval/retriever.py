"""Point-in-time document retrieval. Ranking is cosine distance, then document id."""

from dataclasses import dataclass
from datetime import date, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models import Document
from research.retrieval.embeddings import EMBEDDING_DIMENSIONS, EmbeddingError, EmbeddingProvider
from research.validation.temporal import (
    Eligibility,
    EligibilityReason,
    LookaheadError,
    as_temporal,
    evaluate,
)

MAX_RESULTS = 50


class RetrievalError(Exception):
    """The retrieval query is outside the supported bounds."""


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    decision_timestamp: datetime
    data_cutoff: datetime
    company_id: UUID | None = None
    document_types: tuple[str, ...] | None = None
    filing_date_start: date | None = None
    filing_date_end: date | None = None
    limit: int = 10


@dataclass(frozen=True)
class RetrievalHit:
    document_id: UUID
    company_id: UUID | None
    document_type: str
    accession_number: str | None
    published_at: datetime
    similarity: float


class DocumentRetriever:
    def __init__(self, session: Session, provider: EmbeddingProvider) -> None:
        if provider.dimensions != EMBEDDING_DIMENSIONS:
            raise EmbeddingError("embedding provider width does not match the document column")
        self._session = session
        self._provider = provider

    def retrieve(self, query: RetrievalQuery) -> tuple[RetrievalHit, ...]:
        """Return eligible documents. A future publication is never included."""
        if not 1 <= query.limit <= MAX_RESULTS:
            raise RetrievalError("limit must be from 1 to 50")
        _require_aware(query.decision_timestamp)
        _require_aware(query.data_cutoff)
        if query.decision_timestamp > query.data_cutoff:
            raise LookaheadError(EligibilityReason.DECISION_AFTER_CUTOFF)
        vector = list(self._provider.embed(query.text))
        distance = Document.embedding.cosine_distance(vector)
        statement = select(Document, distance.label("distance")).where(
            Document.embedding.is_not(None),
            Document.published_at.is_not(None),
            Document.published_at <= query.decision_timestamp,
            Document.published_at <= query.data_cutoff,
        )
        if query.company_id is not None:
            statement = statement.where(Document.company_id == query.company_id)
        if query.document_types:
            statement = statement.where(Document.document_type.in_(query.document_types))
        if query.filing_date_start is not None:
            statement = statement.where(Document.filing_date >= query.filing_date_start)
        if query.filing_date_end is not None:
            statement = statement.where(Document.filing_date <= query.filing_date_end)
        statement = statement.order_by(distance.asc(), Document.id.asc()).limit(query.limit)
        hits: list[RetrievalHit] = []
        for document, raw_distance in self._session.execute(statement):
            decision = evaluate(
                as_temporal(document),
                query.decision_timestamp,
                query.data_cutoff,
            )
            if decision.status is not Eligibility.ALLOWED or document.published_at is None:
                continue
            hits.append(
                RetrievalHit(
                    document_id=document.id,
                    company_id=document.company_id,
                    document_type=document.document_type,
                    accession_number=document.accession_number,
                    published_at=document.published_at,
                    similarity=1.0 - float(raw_distance),
                )
            )
        return tuple(hits)


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LookaheadError(EligibilityReason.NAIVE_TIMESTAMP)
