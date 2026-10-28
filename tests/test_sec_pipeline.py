"""SEC ingestion against local fixtures. No live EDGAR requests."""

import os
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from api.models import Document
from data.sec.client import (
    HttpResponse,
    SecClient,
    SecConfigError,
    sec_client_from_env,
)
from data.sec.ingest import IngestBoundsError, IngestRequest, ingest
from data.sec.parse import MalformedFiling, parse_company_tickers
from data.sec.store import MemoryFilingStore, SqlFilingStore, stable_id
from research.validation.temporal import Eligibility, TemporalRecord, evaluate

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sec"
USER_AGENT = "Financial Intelligence Lab tests research@example.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000000123.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000123.json"
MALFORMED_URL = "https://data.sec.gov/submissions/CIK0000000999.json"
TEN_K_URL = "https://www.sec.gov/Archives/edgar/data/123/000000012324000001/acme-20231231.htm"
TEN_Q_URL = "https://www.sec.gov/Archives/edgar/data/123/000000012324000002/acme-20240331.htm"
EIGHT_K_URL = "https://www.sec.gov/Archives/edgar/data/123/000000012324000003/acme-8k.htm"
FUTURE_TEN_K_URL = (
    "https://www.sec.gov/Archives/edgar/data/123/000000012325000001/acme-20241231.htm"
)
TEN_K_HASH = "58831dbfa04d60745853723433acb8b6583ec46b78c57edb37c9ef5388a15fbe"
RETRIEVED_AT = datetime(2026, 1, 15, 12, 0, tzinfo=UTC)
CUTOFF = datetime(2024, 9, 1, tzinfo=UTC)
EARLY_CUTOFF = datetime(2024, 6, 1, tzinfo=UTC)
TEN_K_ACCESSION = "0000000123-24-000001"

ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_sec_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"


class FixtureTransport:
    """Serves the checked-in EDGAR fixtures and records every request."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.user_agents: list[str] = []
        self._files = {
            TICKERS_URL: FIXTURES / "company_tickers.json",
            SUBMISSIONS_URL: FIXTURES / "submissions_CIK0000000123.json",
            FACTS_URL: FIXTURES / "companyfacts_CIK0000000123.json",
            MALFORMED_URL: FIXTURES / "submissions_CIK0000000999.json",
            TEN_K_URL: FIXTURES / "acme-20231231.htm",
            TEN_Q_URL: FIXTURES / "acme-20240331.htm",
            EIGHT_K_URL: FIXTURES / "acme-8k.htm",
        }

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        self.calls.append(url)
        self.user_agents.append(headers["User-Agent"])
        path = self._files.get(url)
        if path is None:
            raise AssertionError(f"unexpected SEC request: {url}")
        return HttpResponse(status=200, body=path.read_bytes(), headers={})


class ScriptedTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = responses
        self.calls = 0
        self.user_agents: list[str] = []

    def get(self, url: str, headers: Mapping[str, str]) -> HttpResponse:
        self.user_agents.append(headers["User-Agent"])
        response = self._responses[self.calls]
        self.calls += 1
        return response


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _request(
    tickers: tuple[str, ...] = ("ACME",),
    forms: tuple[str, ...] = ("10-K",),
    max_filings: int = 1,
    cutoff: datetime = EARLY_CUTOFF,
) -> IngestRequest:
    return IngestRequest(
        tickers=tickers,
        forms=forms,
        max_filings_per_company=max_filings,
        data_cutoff=cutoff,
        retrieved_at=RETRIEVED_AT,
    )


def _client(transport: FixtureTransport | ScriptedTransport) -> SecClient:
    return SecClient(
        user_agent=USER_AGENT,
        transport=transport,
        clock=lambda: 0.0,
        sleep=lambda _seconds: None,
        min_interval=0,
    )


def test_cik_mapping_pads_and_rejects_unknown_tickers() -> None:
    mapping = parse_company_tickers((FIXTURES / "company_tickers.json").read_bytes())
    assert mapping.cik_for("acme") == "0000000123"
    assert mapping.name_for("ACME") == "Acme Industrial Corp"
    with pytest.raises(KeyError):
        mapping.cik_for("ZZZZ")


def test_filing_retrieval_reads_the_fixture_submission_and_document() -> None:
    transport = FixtureTransport()
    result = ingest(_request(), _client(transport), MemoryFilingStore())
    assert result.inserted_accessions == (TEN_K_ACCESSION,)
    assert transport.calls == [TICKERS_URL, SUBMISSIONS_URL, FACTS_URL, TEN_K_URL]
    assert transport.user_agents == [USER_AGENT, USER_AGENT, USER_AGENT, USER_AGENT]
    assert FUTURE_TEN_K_URL not in transport.calls
    assert result.filings[0].content.startswith("ACME FORM 10-K")


def test_duplicate_ingestion_does_not_download_or_insert_again() -> None:
    transport = FixtureTransport()
    store = MemoryFilingStore()
    client = _client(transport)
    first = ingest(_request(), client, store)
    calls_after_first = len(transport.calls)
    second = ingest(_request(), client, store)
    assert first.inserted_accessions == (TEN_K_ACCESSION,)
    assert second.inserted_accessions == ()
    assert second.skipped_accessions == (TEN_K_ACCESSION,)
    assert len(store.filings) == 1
    assert transport.calls[calls_after_first:] == [TICKERS_URL, SUBMISSIONS_URL]
    assert TEN_K_URL not in transport.calls[calls_after_first:]


def test_malformed_filing_is_rejected_without_a_stored_row() -> None:
    transport = FixtureTransport()
    store = MemoryFilingStore()
    with pytest.raises(MalformedFiling):
        ingest(_request(tickers=("MALFORM",)), _client(transport), store)
    assert store.filings == ()
    assert all(not call.endswith(".htm") for call in transport.calls)


def test_metadata_extraction_preserves_accession_dates_and_source_url() -> None:
    filing = ingest(_request(), _client(FixtureTransport()), MemoryFilingStore()).filings[0]
    assert filing.accession_number == TEN_K_ACCESSION
    assert filing.document_type == "10-K"
    assert filing.filing_date == date(2024, 2, 15)
    assert filing.period_start == date(2023, 1, 1)
    assert filing.period_end == date(2023, 12, 31)
    assert filing.published_at == datetime(2024, 2, 15, 21, 5, tzinfo=UTC)
    assert filing.effective_at == datetime(2024, 2, 15, tzinfo=UTC)
    assert filing.retrieved_at == RETRIEVED_AT
    assert filing.source_url == TEN_K_URL
    assert filing.cik == "0000000123"


def test_content_hash_is_sha256_of_the_fixture_body() -> None:
    filing = ingest(_request(), _client(FixtureTransport()), MemoryFilingStore()).filings[0]
    assert filing.content_hash == TEN_K_HASH
    assert (FIXTURES / "acme-20231231.htm").read_bytes() == filing.content.encode()


def test_point_in_time_metadata_excludes_filings_published_after_the_cutoff() -> None:
    transport = FixtureTransport()
    result = ingest(
        _request(forms=("10-K", "10-Q", "8-K"), max_filings=10, cutoff=CUTOFF),
        _client(transport),
        MemoryFilingStore(),
    )
    assert tuple(filing.accession_number for filing in result.filings) == (
        "0000000123-24-000003",
        "0000000123-24-000002",
        TEN_K_ACCESSION,
    )
    assert FUTURE_TEN_K_URL not in transport.calls
    for filing in result.filings:
        decision = evaluate(
            TemporalRecord(
                filing.published_at,
                filing.effective_at,
                filing.period_start,
                filing.period_end,
                filing.retrieved_at,
            ),
            CUTOFF,
            CUTOFF,
        )
        assert decision.status is Eligibility.ALLOWED
        assert filing.published_at <= CUTOFF
    eight_k = result.filings[0]
    assert eight_k.period_start == date(2024, 7, 15)
    assert eight_k.period_end == date(2024, 7, 15)


def test_retry_backs_off_then_returns_the_successful_response() -> None:
    clock = FakeClock()
    transport = ScriptedTransport(
        [
            HttpResponse(status=503, body=b"busy", headers={}),
            HttpResponse(status=429, body=b"slow down", headers={"Retry-After": "2"}),
            HttpResponse(status=200, body=b"{}", headers={}),
        ]
    )
    client = SecClient(
        user_agent=USER_AGENT,
        transport=transport,
        clock=clock.time,
        sleep=clock.sleep,
        min_interval=0,
        initial_backoff=0.5,
    )
    assert client.get_json(TICKERS_URL) == {}
    assert transport.calls == 3
    assert clock.slept == [0.5, 2.0]


def test_rate_limiter_spaces_requests() -> None:
    clock = FakeClock()
    transport = ScriptedTransport(
        [
            HttpResponse(status=200, body=b"{}", headers={}),
            HttpResponse(status=200, body=b"{}", headers={}),
        ]
    )
    client = SecClient(
        user_agent=USER_AGENT,
        transport=transport,
        clock=clock.time,
        sleep=clock.sleep,
        min_interval=0.1,
    )
    client.get_json(TICKERS_URL)
    client.get_json(SUBMISSIONS_URL)
    assert clock.slept == [0.1]


def test_sec_user_agent_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SecConfigError):
        sec_client_from_env(FixtureTransport())
    monkeypatch.setenv("SEC_USER_AGENT", "   ")
    with pytest.raises(SecConfigError):
        sec_client_from_env(FixtureTransport())


def test_ingest_refuses_an_unbounded_request_before_any_download() -> None:
    transport = FixtureTransport()
    with pytest.raises(IngestBoundsError):
        ingest(
            _request(tickers=tuple(f"T{index}" for index in range(26))),
            _client(transport),
            MemoryFilingStore(),
        )
    assert transport.calls == []


def test_repeat_ingest_is_idempotent_in_postgres(migrated_sec_database: None) -> None:
    del migrated_sec_database
    engine = create_engine(TEST_URL)
    transport = FixtureTransport()
    client = _client(transport)
    with Session(engine) as session:
        store = SqlFilingStore(session)
        first = ingest(_request(), client, store)
        session.commit()
        second = ingest(_request(), client, store)
        session.commit()
        count = session.scalar(select(func.count()).select_from(Document))
    engine.dispose()
    assert count == 1
    assert first.filings[0].id == stable_id("document", TEN_K_ACCESSION)
    assert second.skipped_accessions == (TEN_K_ACCESSION,)
    assert second.filings[0].id == first.filings[0].id
    assert second.filings[0].content_hash == TEN_K_HASH


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(sql))
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_sec_database() -> Iterator[None]:
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
