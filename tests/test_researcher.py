"""Researcher agent tests. The language model is a script and no network call is made."""

import json
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from agents.llm import MockLanguageModel, ModelProviderError
from agents.researcher.agent import (
    MalformedModelResponse,
    Researcher,
    TradeRecommendation,
    UnsupportedClaim,
)
from agents.researcher.evidence import EvidenceExcerpt
from api.models import ResearchArtifact
from prompts.researcher import PROMPT_VERSION
from research.validation.temporal import EligibilityReason, LookaheadError
from schemas.researcher import ResearchFindings, ResearchRequest

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_researcher_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"

QUESTION = "Do guidance cuts predict later US equity returns?"
DECISION = datetime(2024, 6, 1, tzinfo=UTC)
CUTOFF = datetime(2024, 6, 1, tzinfo=UTC)
PUBLISHED = datetime(2024, 5, 15, 16, tzinfo=UTC)
DOCUMENT_ID = UUID("00000000-0000-4000-8000-0000000000d1")
OTHER_ID = UUID("00000000-0000-4000-8000-0000000000d2")


class FakeEvidence:
    def __init__(
        self,
        excerpts: tuple[EvidenceExcerpt, ...] = (),
        error: Exception | None = None,
    ) -> None:
        self.calls: list[ResearchRequest] = []
        self._excerpts = excerpts
        self._error = error

    def collect(self, request: ResearchRequest) -> tuple[EvidenceExcerpt, ...]:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return self._excerpts


def _request() -> ResearchRequest:
    return ResearchRequest(
        question=QUESTION,
        decision_timestamp=DECISION,
        data_cutoff=CUTOFF,
    )


def _excerpt(document_id: UUID = DOCUMENT_ID) -> EvidenceExcerpt:
    return EvidenceExcerpt(
        document_id=document_id,
        accession_number="0000000001-24-000001",
        published_at=PUBLISHED,
        excerpt="Revenue guidance was reduced by eight percent.",
    )


def _body(document_id: UUID = DOCUMENT_ID, summary: str | None = None) -> str:
    summary_text = summary or "Guidance cuts are followed by weaker returns in the filing."
    return json.dumps(
        {
            "research_summary": summary_text,
            "known_facts": [
                {
                    "statement": "The filing says revenue guidance was reduced by eight percent.",
                    "evidence_ids": [str(document_id)],
                    "kind": "observed",
                },
                {
                    "statement": "An earlier filing recorded a similar guidance cut.",
                    "evidence_ids": [str(document_id)],
                    "kind": "historical_analogue",
                },
            ],
            "unknowns": ["The retrieved filing does not measure the later return."],
            "mechanisms": ["Investors reprice expected cash flows after a guidance cut."],
            "potential_variables": ["guidance revision percent"],
            "candidate_hypotheses": [
                {
                    "statement": "A guidance cut precedes weaker sector-relative returns.",
                    "role": "primary",
                },
                {
                    "statement": "The return move is explained by the earnings surprise instead.",
                    "role": "alternative",
                },
            ],
            "falsification_conditions": [
                "The relationship disappears after controlling for the earnings surprise."
            ],
        }
    )


def _researcher(
    session: Session,
    content: str = "",
    *,
    excerpts: tuple[EvidenceExcerpt, ...] = (),
    evidence_error: Exception | None = None,
    model_error: Exception | None = None,
) -> tuple[Researcher, FakeEvidence, MockLanguageModel]:
    evidence = FakeEvidence(excerpts, evidence_error)
    model = MockLanguageModel(content, error=model_error)
    return Researcher(session, evidence, model), evidence, model


def _artifact_count(session: Session) -> int:
    count = session.scalar(select(func.count()).select_from(ResearchArtifact))
    assert count is not None
    return count


def test_schema_compliance(session: Session) -> None:
    researcher, _, _ = _researcher(session, _body(), excerpts=(_excerpt(),))
    findings = researcher.investigate(_request())
    assert isinstance(findings, ResearchFindings)
    assert findings.model_name == "mock"
    assert findings.model_version == "test-1"
    assert findings.prompt_version == PROMPT_VERSION
    assert {fact.kind for fact in findings.known_facts} == {"observed", "historical_analogue"}
    assert {item.role for item in findings.candidate_hypotheses} == {"primary", "alternative"}
    assert findings.mechanisms
    assert findings.potential_variables
    assert findings.falsification_conditions


def test_evidence_attribution(session: Session) -> None:
    researcher, _, model = _researcher(session, _body(), excerpts=(_excerpt(),))
    findings = researcher.investigate(_request())
    prompt = model.requests[0].user_prompt
    assert str(DOCUMENT_ID) in prompt
    assert str(OTHER_ID) not in prompt
    assert model.requests[0].prompt_version == PROMPT_VERSION
    assert findings.known_facts[0].evidence_ids == [DOCUMENT_ID]


def test_unsupported_claims_are_rejected(session: Session) -> None:
    unknown, _, _ = _researcher(session, _body(OTHER_ID), excerpts=(_excerpt(),))
    with pytest.raises(UnsupportedClaim):
        unknown.investigate(_request())
    empty, _, empty_model = _researcher(session, _body(), excerpts=())
    with pytest.raises(UnsupportedClaim):
        empty.investigate(_request())
    blocked, _, blocked_model = _researcher(
        session,
        _body(),
        evidence_error=LookaheadError(EligibilityReason.DECISION_AFTER_CUTOFF),
    )
    with pytest.raises(LookaheadError):
        blocked.investigate(_request())
    assert blocked_model.requests == []
    assert empty_model.requests
    assert _artifact_count(session) == 0


def test_malformed_model_response_is_rejected(session: Session) -> None:
    invalid, _, _ = _researcher(session, "not json", excerpts=(_excerpt(),))
    with pytest.raises(MalformedModelResponse):
        invalid.investigate(_request())
    missing, _, _ = _researcher(
        session,
        '{"research_summary": "only one field"}',
        excerpts=(_excerpt(),),
    )
    with pytest.raises(MalformedModelResponse):
        missing.investigate(_request())
    assert _artifact_count(session) == 0


def test_provider_failure_writes_nothing(session: Session) -> None:
    researcher, _, _ = _researcher(
        session,
        excerpts=(_excerpt(),),
        model_error=ModelProviderError("provider unavailable"),
    )
    with pytest.raises(ModelProviderError):
        researcher.investigate(_request())
    assert _artifact_count(session) == 0


def test_trade_recommendation_is_rejected(session: Session) -> None:
    buy, _, _ = _researcher(
        session,
        _body(summary="Investors should buy ACME now."),
        excerpts=(_excerpt(),),
    )
    with pytest.raises(TradeRecommendation):
        buy.investigate(_request())
    coded, _, _ = _researcher(
        session,
        _body(summary="```python\nprint(1)\n```"),
        excerpts=(_excerpt(),),
    )
    with pytest.raises(TradeRecommendation):
        coded.investigate(_request())
    allowed, _, _ = _researcher(
        session,
        _body(summary="The filing describes selling pressure after the guidance cut."),
        excerpts=(_excerpt(),),
    )
    stored = allowed.investigate(_request())
    assert "selling pressure" in stored.research_summary
    assert _artifact_count(session) == 1


def test_artifact_persistence_is_idempotent(session: Session) -> None:
    researcher, _, _ = _researcher(session, _body(), excerpts=(_excerpt(),))
    researcher.investigate(_request())
    first = session.scalars(select(ResearchArtifact)).one()
    researcher.investigate(_request())
    second = session.scalars(select(ResearchArtifact)).one()
    assert first.id == second.id
    assert _artifact_count(session) == 1
    assert first.artifact_type == "RESEARCH_NOTE"
    assert first.experiment_id is None
    assert first.model_name == "mock"
    assert first.model_version == "test-1"
    assert first.prompt_version == PROMPT_VERSION
    body = json.loads(first.content)
    stored_input = ResearchRequest.model_validate(body["input"])
    stored_output = ResearchFindings.model_validate(body["output"])
    assert stored_input.question == QUESTION
    assert stored_input.decision_timestamp == DECISION
    assert stored_input.data_cutoff == CUTOFF
    assert stored_output.prompt_version == PROMPT_VERSION
    assert body["evidence"][0]["document_id"] == str(DOCUMENT_ID)


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(sql))
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_researcher_database() -> Iterator[None]:
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
def session(migrated_researcher_database: None) -> Iterator[Session]:
    del migrated_researcher_database
    engine = create_engine(TEST_URL)
    with Session(engine) as db:
        try:
            yield db
        finally:
            db.rollback()
    engine.dispose()
