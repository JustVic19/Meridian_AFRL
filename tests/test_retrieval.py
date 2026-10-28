"""Document retrieval against Postgres. The embedding provider is the deterministic mock."""

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from api.models import Company, Document
from research.retrieval.embeddings import MockEmbeddingProvider, assign_embedding
from research.retrieval.retriever import DocumentRetriever, RetrievalQuery
from research.validation.temporal import LookaheadError

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_retrieval_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"

DECISION = datetime(2024, 6, 1, tzinfo=UTC)
CUTOFF = datetime(2024, 6, 1, tzinfo=UTC)
PUBLISHED = datetime(2024, 5, 15, 16, 0, tzinfo=UTC)
RETRIEVED = datetime(2024, 5, 16, tzinfo=UTC)
PERIOD_START = date(2024, 1, 1)
PERIOD_END = date(2024, 3, 31)

COMPANY_A = UUID("00000000-0000-4000-8000-0000000000a1")
COMPANY_B = UUID("00000000-0000-4000-8000-0000000000b2")


def _company(company_id: UUID, ticker: str, cik: str) -> Company:
    return Company(
        id=company_id,
        ticker=ticker,
        cik=cik,
        name=f"{ticker} Industries",
        exchange="NYSE",
    )


def _document(
    document_id: UUID,
    company_id: UUID,
    content: str,
    accession: str,
    *,
    document_type: str = "10-K",
    filing_date: date = date(2024, 5, 15),
    published_at: datetime = PUBLISHED,
) -> Document:
    return Document(
        id=document_id,
        company_id=company_id,
        document_type=document_type,
        accession_number=accession,
        filing_date=filing_date,
        period_start=PERIOD_START,
        period_end=PERIOD_END,
        published_at=published_at,
        effective_at=published_at,
        retrieved_at=RETRIEVED,
        source_url=f"https://www.sec.gov/Archives/edgar/data/{accession}",
        content=content,
        content_hash=f"hash-{accession}",
    )


def _query(**overrides: object) -> RetrievalQuery:
    values: dict[str, object] = {
        "text": "revenue guidance",
        "decision_timestamp": DECISION,
        "data_cutoff": CUTOFF,
    }
    values.update(overrides)
    return RetrievalQuery(**values)  # type: ignore[arg-type]


def test_semantic_retrieval_ranks_shared_tokens_first(session: Session) -> None:
    provider = MockEmbeddingProvider()
    guidance_id = UUID("00000000-0000-4000-8000-000000000101")
    dividend_id = UUID("00000000-0000-4000-8000-000000000102")
    session.add(_company(COMPANY_A, "ACME", "0000000101"))
    session.add(
        _document(
            guidance_id,
            COMPANY_A,
            "revenue guidance reduced",
            "0000000101-24-000001",
        )
    )
    session.add(
        _document(
            dividend_id,
            COMPANY_A,
            "quarterly dividend increased",
            "0000000101-24-000002",
        )
    )
    session.flush()
    for document_id in (guidance_id, dividend_id):
        assign_embedding(session, session.get(Document, document_id), provider)
    hits = DocumentRetriever(session, provider).retrieve(_query())
    assert [hit.document_id for hit in hits] == [guidance_id, dividend_id]
    assert hits[0].similarity > hits[1].similarity


def test_company_filter_keeps_one_company(session: Session) -> None:
    provider = MockEmbeddingProvider()
    acme_id = UUID("00000000-0000-4000-8000-000000000201")
    beta_id = UUID("00000000-0000-4000-8000-000000000202")
    session.add(_company(COMPANY_A, "ACME", "0000000201"))
    session.add(_company(COMPANY_B, "BETA", "0000000202"))
    session.add(
        _document(
            acme_id,
            COMPANY_A,
            "revenue guidance reduced for acme",
            "0000000201-24-000001",
        )
    )
    session.add(
        _document(
            beta_id,
            COMPANY_B,
            "revenue guidance reduced for beta",
            "0000000202-24-000001",
        )
    )
    session.flush()
    for document_id in (acme_id, beta_id):
        assign_embedding(session, session.get(Document, document_id), provider)
    retriever = DocumentRetriever(session, provider)
    both = retriever.retrieve(_query())
    filtered = retriever.retrieve(_query(company_id=COMPANY_A))
    assert {hit.document_id for hit in both} == {acme_id, beta_id}
    assert [hit.document_id for hit in filtered] == [acme_id]
    assert filtered[0].company_id == COMPANY_A


def test_temporal_type_and_date_filters_reject_later_publication(session: Session) -> None:
    provider = MockEmbeddingProvider()
    included_id = UUID("00000000-0000-4000-8000-000000000301")
    future_id = UUID("00000000-0000-4000-8000-000000000302")
    early_id = UUID("00000000-0000-4000-8000-000000000303")
    other_type_id = UUID("00000000-0000-4000-8000-000000000304")
    session.add(_company(COMPANY_A, "ACME", "0000000301"))
    session.add(
        _document(
            included_id,
            COMPANY_A,
            "revenue guidance reduced",
            "0000000301-24-000001",
        )
    )
    session.add(
        _document(
            future_id,
            COMPANY_A,
            "revenue guidance",
            "0000000301-24-000002",
            filing_date=date(2024, 5, 20),
            published_at=datetime(2024, 7, 1, tzinfo=UTC),
        )
    )
    session.add(
        _document(
            early_id,
            COMPANY_A,
            "revenue guidance reduced earlier",
            "0000000301-23-000001",
            filing_date=date(2023, 1, 15),
            published_at=datetime(2023, 2, 1, tzinfo=UTC),
        )
    )
    session.add(
        _document(
            other_type_id,
            COMPANY_A,
            "revenue guidance reduced event",
            "0000000301-24-000003",
            document_type="8-K",
        )
    )
    session.flush()
    for document_id in (included_id, future_id, early_id, other_type_id):
        assign_embedding(session, session.get(Document, document_id), provider)
    hits = DocumentRetriever(session, provider).retrieve(
        _query(
            document_types=("10-K",),
            filing_date_start=date(2024, 5, 1),
            filing_date_end=date(2024, 5, 31),
        )
    )
    assert [hit.document_id for hit in hits] == [included_id]


def test_results_are_deterministic(session: Session) -> None:
    provider = MockEmbeddingProvider()
    guidance_id = UUID("00000000-0000-4000-8000-000000000401")
    dividend_id = UUID("00000000-0000-4000-8000-000000000402")
    session.add(_company(COMPANY_A, "ACME", "0000000401"))
    session.add(
        _document(
            guidance_id,
            COMPANY_A,
            "revenue guidance reduced",
            "0000000401-24-000001",
        )
    )
    session.add(
        _document(
            dividend_id,
            COMPANY_A,
            "quarterly dividend increased",
            "0000000401-24-000002",
        )
    )
    session.flush()
    for document_id in (guidance_id, dividend_id):
        assign_embedding(session, session.get(Document, document_id), provider)
    retriever = DocumentRetriever(session, provider)

    def ranked(experiment_timestamp: datetime) -> list[tuple[UUID, float]]:
        del experiment_timestamp
        return [(hit.document_id, hit.similarity) for hit in retriever.retrieve(_query())]

    assert ranked(datetime(2024, 3, 1, tzinfo=UTC)) == ranked(datetime(2026, 1, 1, tzinfo=UTC))


def test_missing_embedding_is_omitted(session: Session) -> None:
    provider = MockEmbeddingProvider()
    closer_id = UUID("00000000-0000-4000-8000-000000000501")
    embedded_id = UUID("00000000-0000-4000-8000-000000000502")
    session.add(_company(COMPANY_A, "ACME", "0000000501"))
    session.add(_document(closer_id, COMPANY_A, "revenue guidance", "0000000501-24-000001"))
    session.add(
        _document(
            embedded_id,
            COMPANY_A,
            "quarterly dividend increased",
            "0000000501-24-000002",
        )
    )
    session.flush()
    assign_embedding(session, session.get(Document, embedded_id), provider)
    hits = DocumentRetriever(session, provider).retrieve(_query())
    assert [hit.document_id for hit in hits] == [embedded_id]


def test_duplicate_vectors_stay_ordered_and_assignment_does_not_insert(
    session: Session,
) -> None:
    provider = MockEmbeddingProvider()
    first_id = UUID("00000000-0000-4000-8000-000000000601")
    second_id = UUID("00000000-0000-4000-8000-000000000602")
    session.add(_company(COMPANY_A, "ACME", "0000000601"))
    session.add(_document(first_id, COMPANY_A, "revenue guidance alpha", "0000000601-24-000001"))
    session.add(_document(second_id, COMPANY_A, "revenue guidance beta", "0000000601-24-000002"))
    session.flush()
    first = session.get(Document, first_id)
    second = session.get(Document, second_id)
    assert first is not None and second is not None
    assign_embedding(session, first, provider)
    assign_embedding(session, first, provider)
    assign_embedding(session, second, provider)
    assert first.embedding is not None
    second.embedding = list(first.embedding)
    session.flush()
    hits = DocumentRetriever(session, provider).retrieve(_query())
    assert [hit.document_id for hit in hits] == [first_id, second_id]
    assert hits[0].similarity == hits[1].similarity
    assert session.scalar(select(func.count()).select_from(Document)) == 2


def test_future_information_is_rejected(session: Session) -> None:
    provider = MockEmbeddingProvider()
    future_id = UUID("00000000-0000-4000-8000-000000000701")
    allowed_id = UUID("00000000-0000-4000-8000-000000000702")
    session.add(_company(COMPANY_A, "ACME", "0000000701"))
    session.add(
        _document(
            future_id,
            COMPANY_A,
            "revenue guidance",
            "0000000701-24-000001",
            published_at=datetime(2024, 7, 1, tzinfo=UTC),
        )
    )
    session.add(
        _document(
            allowed_id,
            COMPANY_A,
            "revenue guidance reduced",
            "0000000701-24-000002",
        )
    )
    session.flush()
    for document_id in (future_id, allowed_id):
        assign_embedding(session, session.get(Document, document_id), provider)
    retriever = DocumentRetriever(session, provider)
    hits = retriever.retrieve(_query())
    assert [hit.document_id for hit in hits] == [allowed_id]
    with pytest.raises(LookaheadError, match="DECISION_AFTER_CUTOFF"):
        retriever.retrieve(_query(decision_timestamp=datetime(2024, 8, 1, tzinfo=UTC)))


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(sql))
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_retrieval_database() -> Iterator[None]:
    _admin(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{TEST_DB}' AND pid <> pg_backend_pid()"
    )
    _admin(f"DROP DATABASE IF EXISTS {TEST_DB}")
    _admin(f"CREATE DATABASE {TEST_DB}")
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_URL
    try:
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        command.upgrade(config, "head")
        yield
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
        _admin(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{TEST_DB}' AND pid <> pg_backend_pid()"
        )
        _admin(f"DROP DATABASE IF EXISTS {TEST_DB}")


@pytest.fixture
def session(migrated_retrieval_database: None) -> Iterator[Session]:
    del migrated_retrieval_database
    engine = create_engine(TEST_URL)
    with Session(engine) as db:
        try:
            yield db
        finally:
            db.rollback()
    engine.dispose()
