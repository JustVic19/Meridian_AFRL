"""Evidence passed to the Researcher. Excerpts come from cutoff-safe retrieval hits."""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from api.models import Document
from research.retrieval.retriever import DocumentRetriever, RetrievalQuery
from schemas.researcher import ResearchRequest

EXCERPT_CHARACTERS = 500


@dataclass(frozen=True)
class EvidenceExcerpt:
    document_id: UUID
    accession_number: str | None
    published_at: datetime
    excerpt: str


class EvidenceSource(Protocol):
    def collect(self, request: ResearchRequest) -> tuple[EvidenceExcerpt, ...]:
        """Return documents the researcher is allowed to cite."""


class DatabaseEvidence:
    """Load a short excerpt for each document the retriever already judged eligible."""

    def __init__(self, session: Session, retriever: DocumentRetriever) -> None:
        self._session = session
        self._retriever = retriever

    def collect(self, request: ResearchRequest) -> tuple[EvidenceExcerpt, ...]:
        hits = self._retriever.retrieve(
            RetrievalQuery(
                text=request.question,
                decision_timestamp=request.decision_timestamp,
                data_cutoff=request.data_cutoff,
            )
        )
        excerpts: list[EvidenceExcerpt] = []
        for hit in hits:
            document = self._session.get(Document, hit.document_id)
            content = "" if document is None or document.content is None else document.content
            excerpts.append(
                EvidenceExcerpt(
                    document_id=hit.document_id,
                    accession_number=hit.accession_number,
                    published_at=hit.published_at,
                    excerpt=content[:EXCERPT_CHARACTERS],
                )
            )
        return tuple(excerpts)
