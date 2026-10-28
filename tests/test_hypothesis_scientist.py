"""HypothesisScientist tests. The language model is a script and no network call is made."""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from agents.hypothesis.agent import HypothesisScientist, InvalidHypothesis
from agents.llm import MockLanguageModel
from api.models import Hypothesis, ResearchArtifact
from prompts.hypothesis import PROMPT_VERSION
from schemas.hypothesis import HypothesisRequest, SupportingEvidence
from schemas.research import HypothesisStatus
from schemas.researcher import (
    CandidateHypothesis,
    KnownFact,
    ResearchFindings,
)

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_hypothesis_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"

QUESTION = "Do guidance cuts predict later US equity returns?"
DOCUMENT_ID = UUID("00000000-0000-4000-8000-0000000000d1")
VALID_STATEMENT = (
    "Among US-listed companies above $2B market capitalization, negative forward "
    "revenue guidance revisions exceeding 5% predict negative sector-relative "
    "returns over the following 60 trading days."
)
REVISED_STATEMENT = (
    "Among US-listed companies above $2B market capitalization, negative forward "
    "revenue guidance revisions exceeding 8% predict negative sector-relative "
    "returns over the following 60 trading days."
)


def _findings() -> ResearchFindings:
    return ResearchFindings(
        research_summary="A filing reported an eight percent guidance reduction.",
        known_facts=[
            KnownFact(
                statement="The filing says revenue guidance was reduced by eight percent.",
                evidence_ids=[DOCUMENT_ID],
                kind="observed",
            ),
            KnownFact(
                statement="An earlier filing recorded a similar guidance cut.",
                evidence_ids=[DOCUMENT_ID],
                kind="historical_analogue",
            ),
        ],
        unknowns=["The filing does not measure the later return."],
        mechanisms=["Investors reprice expected cash flows after a guidance cut."],
        potential_variables=["guidance revision percent"],
        candidate_hypotheses=[
            CandidateHypothesis(
                statement="A guidance cut precedes weaker sector-relative returns.",
                role="primary",
            ),
            CandidateHypothesis(
                statement="The return move is explained by the earnings surprise instead.",
                role="alternative",
            ),
        ],
        falsification_conditions=[
            "The relationship disappears after controlling for the earnings surprise."
        ],
        model_name="mock",
        model_version="test-1",
        prompt_version="researcher-1",
    )


def _request(parent_id: UUID | None = None) -> HypothesisRequest:
    return HypothesisRequest(
        question=QUESTION,
        findings=_findings(),
        evidence=[
            SupportingEvidence(
                document_id=DOCUMENT_ID,
                excerpt="Revenue guidance was reduced by eight percent.",
                accession_number="0000000001-24-000001",
            )
        ],
        parent_hypothesis_id=parent_id,
    )


def _spec(statement: str, threshold: float, *, include_event: bool = True) -> str:
    payload: dict[str, object] = {
        "statement": statement,
        "universe": {"asset_class": "US_EQUITY", "min_market_cap": 2_000_000_000},
        "entry_rule": {"timing": "NEXT_MARKET_OPEN"},
        "exit_rule": {"holding_period_days": 60},
        "benchmark": "SECTOR",
        "holding_period": 60,
        "mechanism": "Investors reprice expected cash flows after the guidance cut.",
        "falsification_conditions": [
            "The relationship disappears after controlling for the earnings surprise."
        ],
    }
    if include_event:
        payload["event_definition"] = {"type": "GUIDANCE_REVISION", "threshold": threshold}
    return json.dumps(payload)


def _count(session: Session, model: type[Hypothesis] | type[ResearchArtifact]) -> int:
    count = session.scalar(select(func.count()).select_from(model))
    assert count is not None
    return count


def test_valid_hypothesis_is_precise(session: Session) -> None:
    scientist = HypothesisScientist(session, MockLanguageModel(_spec(VALID_STATEMENT, -0.05)))
    formalized = scientist.formalize(_request())
    specification = formalized.specification
    assert formalized.version == 1
    assert formalized.parent_hypothesis_id is None
    assert specification.statement == VALID_STATEMENT
    assert specification.universe.asset_class == "US_EQUITY"
    assert specification.universe.min_market_cap == 2_000_000_000
    assert specification.event_definition.type == "GUIDANCE_REVISION"
    assert specification.event_definition.threshold == -0.05
    assert specification.entry_rule.timing == "NEXT_MARKET_OPEN"
    assert specification.exit_rule.holding_period_days == 60
    assert specification.benchmark == "SECTOR"
    assert specification.holding_period == 60
    assert specification.mechanism
    assert specification.falsification_conditions
    assert specification.prompt_version == PROMPT_VERSION
    stored = session.get(Hypothesis, formalized.id)
    assert stored is not None
    assert stored.status == HypothesisStatus.DRAFT.value
    assert stored.expected_horizon == 60


def test_vague_hypothesis_is_rejected(session: Session) -> None:
    scientist = HypothesisScientist(
        session,
        MockLanguageModel(_spec("Guidance cuts are bad.", -0.05)),
    )
    with pytest.raises(InvalidHypothesis, match="vague hypothesis"):
        scientist.formalize(_request())
    assert _count(session, Hypothesis) == 0


def test_missing_event_definition_is_rejected(session: Session) -> None:
    scientist = HypothesisScientist(
        session,
        MockLanguageModel(_spec(VALID_STATEMENT, -0.05, include_event=False)),
    )
    with pytest.raises(InvalidHypothesis, match="missing event definition"):
        scientist.formalize(_request())
    assert _count(session, Hypothesis) == 0


def test_missing_falsification_conditions_are_rejected(session: Session) -> None:
    payload = json.loads(_spec(VALID_STATEMENT, -0.05))
    payload["falsification_conditions"] = []
    scientist = HypothesisScientist(session, MockLanguageModel(json.dumps(payload)))
    with pytest.raises(InvalidHypothesis, match="missing falsification conditions"):
        scientist.formalize(_request())
    assert _count(session, Hypothesis) == 0


def test_versioning_keeps_the_parent_and_links_the_child(session: Session) -> None:
    parent = HypothesisScientist(
        session,
        MockLanguageModel(_spec(VALID_STATEMENT, -0.05)),
    ).formalize(_request())
    parent_statement = session.get(Hypothesis, parent.id)
    assert parent_statement is not None
    child = HypothesisScientist(
        session,
        MockLanguageModel(_spec(REVISED_STATEMENT, -0.08)),
    ).formalize(_request(parent.id))
    assert parent.version == 1
    assert child.version == 2
    assert child.parent_hypothesis_id == parent.id
    assert child.id != parent.id
    unchanged = session.get(Hypothesis, parent.id)
    assert unchanged is not None
    assert unchanged.statement == parent_statement.statement
    assert unchanged.parent_hypothesis_id is None
    stored_child = session.get(Hypothesis, child.id)
    assert stored_child is not None
    assert stored_child.parent_hypothesis_id == parent.id
    assert _count(session, Hypothesis) == 2


def test_artifact_persistence_does_not_duplicate(session: Session) -> None:
    scientist = HypothesisScientist(session, MockLanguageModel(_spec(VALID_STATEMENT, -0.05)))
    request = _request()
    first = scientist.formalize(request)
    second = scientist.formalize(request)
    assert first.id == second.id
    assert _count(session, Hypothesis) == 1
    assert _count(session, ResearchArtifact) == 1
    artifact = session.scalars(select(ResearchArtifact)).one()
    assert artifact.artifact_type == "HYPOTHESIS"
    assert artifact.experiment_id is None
    assert artifact.model_name == "mock"
    assert artifact.model_version == "test-1"
    assert artifact.prompt_version == PROMPT_VERSION
    body = json.loads(artifact.content)
    assert body["input"]["question"] == QUESTION
    assert body["output"]["statement"] == VALID_STATEMENT
    assert body["output"]["event_definition"]["threshold"] == -0.05
    assert body["input"]["evidence"][0]["document_id"] == str(DOCUMENT_ID)


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(sql))
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_hypothesis_database() -> Iterator[None]:
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
def session(migrated_hypothesis_database: None) -> Iterator[Session]:
    del migrated_hypothesis_database
    engine = create_engine(TEST_URL)
    with Session(engine) as db:
        try:
            yield db
        finally:
            db.rollback()
    engine.dispose()
