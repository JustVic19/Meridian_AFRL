"""QuantAgent tests. The language model is a script, and generated code is never run."""

import json
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

from agents.llm import MockLanguageModel
from agents.quant.agent import QuantAgent, QuantRejected
from api.models import Experiment, Hypothesis, ResearchArtifact
from prompts.quant import PROMPT_VERSION
from research.validation.specification import SpecificationReason, SpecificationRejected
from schemas.experiment_spec import DataRequirement, DatasetCatalog, ExperimentSpecification
from schemas.hypothesis import (
    EntryRule,
    EventDefinition,
    ExitRule,
    FormalizedHypothesis,
    HypothesisSpecification,
    UniverseDefinition,
)
from schemas.research import HypothesisStatus

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_quant_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"

HYPOTHESIS_ID = UUID("00000000-0000-4000-8000-0000000000e1")
DATASET = "prices-2020-01"
CODE = "a1b2c3d"
STATEMENT = (
    "Among US-listed companies above $2B market capitalization, negative forward "
    "revenue guidance revisions exceeding 5% predict negative sector-relative "
    "returns over the following 60 trading days."
)
RESEARCH_CODE = """
def research_signal(close: float, threshold: float) -> bool:
    return close < threshold

raise RuntimeError("executed")
"""


def _hypothesis() -> FormalizedHypothesis:
    return FormalizedHypothesis(
        id=HYPOTHESIS_ID,
        version=1,
        parent_hypothesis_id=None,
        specification=HypothesisSpecification(
            statement=STATEMENT,
            universe=UniverseDefinition(asset_class="US_EQUITY", min_market_cap=2_000_000_000),
            event_definition=EventDefinition(type="GUIDANCE_REVISION", threshold=-0.05),
            entry_rule=EntryRule(timing="NEXT_MARKET_OPEN"),
            exit_rule=ExitRule(holding_period_days=60),
            benchmark="SECTOR",
            holding_period=60,
            mechanism="Investors reprice expected cash flows after the guidance cut.",
            falsification_conditions=[
                "The relationship disappears after controlling for the earnings surprise."
            ],
            model_name="mock",
            model_version="test-1",
            prompt_version="hypothesis-scientist-1",
        ),
    )


def _specification(**overrides: object) -> ExperimentSpecification:
    values: dict[str, object] = {
        "universe": UniverseDefinition(asset_class="US_EQUITY", min_market_cap=2_000_000_000),
        "event": EventDefinition(type="GUIDANCE_REVISION", threshold=-0.05),
        "entry": EntryRule(timing="NEXT_MARKET_OPEN"),
        "exit": ExitRule(holding_period_days=60),
        "benchmark": "SECTOR",
        "data_requirements": [DataRequirement(name="prices", fields=["close", "published_at"])],
        "required_fields": ["close", "published_at"],
        "transaction_cost_bps": 5,
        "slippage_bps": 2,
        "random_seed": 7,
        "dataset_version": DATASET,
        "code_version": CODE,
        "data_cutoff": datetime(2020, 1, 10, 21, tzinfo=UTC),
    }
    values.update(overrides)
    return ExperimentSpecification(**values)  # type: ignore[arg-type]


def _catalog() -> DatasetCatalog:
    return DatasetCatalog(
        dataset_version=DATASET,
        start=date(2020, 1, 1),
        end=date(2020, 1, 10),
        fields=["close", "published_at"],
        published_at=datetime(2020, 1, 10, 18, tzinfo=UTC),
    )


def _completion(code: str = RESEARCH_CODE, dependencies: list[str] | None = None) -> str:
    return json.dumps(
        {
            "research_code": code,
            "dependencies": ["pandas"] if dependencies is None else dependencies,
            "expected_outputs": ["signal", "forward_return"],
        }
    )


def _store_hypothesis(session: Session) -> None:
    spec = _hypothesis().specification
    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="GUIDANCE_REVISION 60d",
            statement=spec.statement,
            universe=spec.universe.model_dump(mode="json"),
            mechanism=spec.mechanism,
            entry_rule=spec.entry_rule.model_dump(mode="json"),
            exit_rule=spec.exit_rule.model_dump(mode="json"),
            expected_horizon=spec.holding_period,
            status=HypothesisStatus.DRAFT.value,
        )
    )
    session.flush()


def _count(session: Session, model: type[Experiment] | type[ResearchArtifact]) -> int:
    count = session.scalar(select(func.count()).select_from(model))
    assert count is not None
    return count


def test_valid_code_is_generated_and_not_executed(session: Session) -> None:
    _store_hypothesis(session)
    model = MockLanguageModel(_completion())
    program = QuantAgent(session, model).generate(_hypothesis(), _specification(), _catalog())
    assert program.research_code == RESEARCH_CODE
    assert program.dependencies == ["pandas"]
    assert program.expected_outputs == ["signal", "forward_return"]
    assert program.execution_metadata.train_start == date(2020, 1, 1)
    assert program.execution_metadata.train_end == date(2020, 1, 6)
    assert program.execution_metadata.validation_start == date(2020, 1, 7)
    assert program.execution_metadata.test_end == date(2020, 1, 10)
    assert program.execution_metadata.random_seed == 7
    stored = session.get(Experiment, program.experiment_id)
    assert stored is not None
    assert stored.metrics is None
    assert stored.status == "CODE_GENERATED"


def test_invalid_specification_is_rejected_before_generation(session: Session) -> None:
    _store_hypothesis(session)
    model = MockLanguageModel(_completion())
    with pytest.raises(SpecificationRejected) as rejected:
        QuantAgent(session, model).generate(
            _hypothesis(),
            _specification(benchmark=""),
            _catalog(),
        )
    assert rejected.value.reasons == (SpecificationReason.MISSING_BENCHMARK,)
    assert model.requests == []
    assert _count(session, Experiment) == 0
    assert _count(session, ResearchArtifact) == 0


def test_artifact_persistence_keeps_one_code_record(session: Session) -> None:
    _store_hypothesis(session)
    agent = QuantAgent(session, MockLanguageModel(_completion()))
    hypothesis = _hypothesis()
    specification = _specification()
    catalog = _catalog()
    first = agent.generate(hypothesis, specification, catalog)
    second = agent.generate(hypothesis, specification, catalog)
    assert first.experiment_id == second.experiment_id
    assert _count(session, Experiment) == 1
    assert _count(session, ResearchArtifact) == 1
    artifact = session.scalars(select(ResearchArtifact)).one()
    assert artifact.artifact_type == "CODE"
    assert artifact.experiment_id == first.experiment_id
    body = json.loads(artifact.content)
    assert body["research_code"] == RESEARCH_CODE
    assert body["code_version"] == CODE


def test_version_metadata_comes_from_the_provider_and_specification(session: Session) -> None:
    _store_hypothesis(session)
    program = QuantAgent(session, MockLanguageModel(_completion())).generate(
        _hypothesis(),
        _specification(),
        _catalog(),
    )
    assert program.model_name == "mock"
    assert program.model_version == "test-1"
    assert program.prompt_version == PROMPT_VERSION
    assert program.code_version == CODE
    artifact = session.scalars(select(ResearchArtifact)).one()
    assert artifact.model_name == "mock"
    assert artifact.model_version == "test-1"
    assert artifact.prompt_version == PROMPT_VERSION
    experiment = session.get(Experiment, program.experiment_id)
    assert experiment is not None
    assert experiment.code_version == CODE
    assert experiment.hypothesis_id == HYPOTHESIS_ID


def test_production_access_is_refused(session: Session) -> None:
    _store_hypothesis(session)
    shell_code = "import subprocess\nsubprocess.run([])\n"
    agent = QuantAgent(session, MockLanguageModel(_completion(shell_code)))
    assert agent.boundary.network is None
    assert agent.boundary.production_database_write is None
    assert agent.boundary.production_filesystem is None
    assert agent.boundary.shell is None
    with pytest.raises(QuantRejected, match="production access"):
        agent.generate(_hypothesis(), _specification(), _catalog())
    assert _count(session, Experiment) == 0
    assert _count(session, ResearchArtifact) == 0
    denied = QuantAgent(
        session,
        MockLanguageModel(_completion(dependencies=["requests"])),
    )
    with pytest.raises(QuantRejected, match="production access"):
        denied.generate(_hypothesis(), _specification(), _catalog())
    assert _count(session, Experiment) == 0


def test_credential_access_is_refused(session: Session) -> None:
    _store_hypothesis(session)
    model = MockLanguageModel(_completion('key = os.environ["BROKER_API_KEY"]\n'))
    agent = QuantAgent(session, model)
    assert agent.boundary.broker_credentials is None
    with pytest.raises(QuantRejected, match="credential access"):
        agent.generate(_hypothesis(), _specification(), _catalog())
    assert model.requests
    prompt = model.requests[0].user_prompt + model.requests[0].system_prompt
    for forbidden in ("BROKER_API_KEY", "password", "DATABASE_URL", "api_key"):
        assert forbidden not in prompt
    assert _count(session, Experiment) == 0
    assert _count(session, ResearchArtifact) == 0


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(sql))
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_quant_database() -> Iterator[None]:
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
def session(migrated_quant_database: None) -> Iterator[Session]:
    del migrated_quant_database
    engine = create_engine(TEST_URL)
    with Session(engine) as db:
        try:
            yield db
        finally:
            db.rollback()
    engine.dispose()
