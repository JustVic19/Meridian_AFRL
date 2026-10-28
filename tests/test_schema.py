"""Deterministic migration tests against a throwaway database."""

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from api.models import (
    Base,
    Company,
    Document,
    Experiment,
    Hypothesis,
    Observation,
    ResearchArtifact,
    Strategy,
)
from schemas.research import (
    ArtifactType,
    CompanyRecord,
    DocumentRecord,
    ExperimentRecord,
    HypothesisRecord,
    HypothesisStatus,
    ObservationRecord,
    ResearchArtifactRecord,
    StrategyRecord,
)

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_schema_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"

CREATED = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
PUBLISHED = datetime(2022, 5, 4, 20, 0, tzinfo=UTC)
DETECTED = datetime(2022, 5, 4, 21, 0, tzinfo=UTC)
STARTED = datetime(2024, 2, 1, 15, 0, tzinfo=UTC)
COMPLETED = datetime(2024, 3, 1, 15, 0, tzinfo=UTC)

COMPANY_ID = UUID("00000000-0000-4000-8000-000000000001")
DOCUMENT_ID = UUID("00000000-0000-4000-8000-000000000002")
OBSERVATION_ID = UUID("00000000-0000-4000-8000-000000000003")
HYPOTHESIS_ID = UUID("00000000-0000-4000-8000-000000000004")
CHILD_HYPOTHESIS_ID = UUID("00000000-0000-4000-8000-000000000005")
EXPERIMENT_ID = UUID("00000000-0000-4000-8000-000000000006")
CHILD_EXPERIMENT_ID = UUID("00000000-0000-4000-8000-000000000007")
ARTIFACT_ID = UUID("00000000-0000-4000-8000-000000000008")
STRATEGY_ID = UUID("00000000-0000-4000-8000-000000000009")
DEFAULT_STRATEGY_ID = UUID("00000000-0000-4000-8000-000000000010")

RESEARCH_TABLES = {
    "companies",
    "documents",
    "observations",
    "hypotheses",
    "experiments",
    "research_artifacts",
    "strategies",
}
REQUIRED_INDEXES = {
    "uq_companies_cik",
    "ix_companies_ticker",
    "uq_documents_content_hash",
    "ix_documents_company_id",
    "ix_documents_published_at",
    "ix_documents_accession_number",
    "ix_observations_company_id",
    "ix_observations_detected_at",
    "ix_observations_published_at",
    "ix_hypotheses_parent_hypothesis_id",
    "ix_hypotheses_status",
    "ix_experiments_hypothesis_id",
    "ix_experiments_parent_experiment_id",
    "ix_experiments_status",
    "ix_research_artifacts_experiment_id",
    "ix_research_artifacts_artifact_type",
    "ix_strategies_experiment_id",
}
REQUIRED_CONSTRAINTS = {
    "ck_hypotheses_status",
    "ck_hypotheses_parent_not_self",
    "ck_experiments_train_dates",
    "ck_experiments_validation_dates",
    "ck_experiments_test_dates",
    "ck_experiments_parent_not_self",
    "ck_research_artifacts_artifact_type",
    "ck_strategies_version",
    "fk_documents_company_id",
    "fk_observations_company_id",
    "fk_hypotheses_parent_hypothesis_id",
    "fk_experiments_hypothesis_id",
    "fk_experiments_parent_experiment_id",
    "fk_research_artifacts_experiment_id",
    "fk_strategies_experiment_id",
}


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        conn.execute(text(sql))
    engine.dispose()


def _recreate_database() -> None:
    _admin(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{TEST_DB}' AND pid <> pg_backend_pid()"
    )
    _admin(f"DROP DATABASE IF EXISTS {TEST_DB}")
    _admin(f"CREATE DATABASE {TEST_DB}")


def _drop_database() -> None:
    _admin(
        "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
        f"WHERE datname = '{TEST_DB}' AND pid <> pg_backend_pid()"
    )
    _admin(f"DROP DATABASE IF EXISTS {TEST_DB}")


def _alembic(direction: str) -> None:
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = TEST_URL
    try:
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        if direction == "upgrade":
            command.upgrade(config, "head")
        else:
            command.downgrade(config, "base")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


def _catalog() -> tuple[list[str], list[tuple[str, str, str]], list[tuple[str, str]], str | None]:
    engine = create_engine(TEST_URL)
    with engine.connect() as conn:
        tables = list(
            conn.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
                    "ORDER BY table_name"
                )
            ).scalars()
        )
        constraints = [
            (str(row[0]), str(row[1]), str(row[2]))
            for row in conn.execute(
                text(
                    "SELECT conrelid::regclass::text, conname, pg_get_constraintdef(oid) "
                    "FROM pg_constraint WHERE connamespace = 'public'::regnamespace "
                    "ORDER BY 1, 2"
                )
            )
        ]
        indexes = [
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                text(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE schemaname = 'public' ORDER BY indexname"
                )
            )
        ]
        trigger = conn.execute(
            text(
                "SELECT tgname FROM pg_trigger "
                "WHERE tgname = 'experiments_immutable_when_finalized' AND NOT tgisinternal"
            )
        ).scalar_one_or_none()
    engine.dispose()
    return tables, constraints, indexes, None if trigger is None else str(trigger)


@pytest.fixture(scope="module")
def migrated_database() -> Iterator[str]:
    _recreate_database()
    _alembic("upgrade")
    try:
        yield TEST_URL
    finally:
        _drop_database()


@pytest.fixture
def session(migrated_database: str) -> Iterator[Session]:
    engine = create_engine(migrated_database)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        try:
            yield db
        finally:
            db.rollback()
    engine.dispose()


def test_upgrade_creates_schema(migrated_database: str) -> None:
    tables, constraints, indexes, trigger = _catalog()
    assert RESEARCH_TABLES <= set(tables)
    constraint_names = {name for _, name, _ in constraints}
    index_names = {name for name, _ in indexes}
    assert REQUIRED_CONSTRAINTS <= constraint_names
    assert REQUIRED_INDEXES <= index_names
    assert trigger == "experiments_immutable_when_finalized"
    engine = create_engine(migrated_database)
    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        db_columns = {column["name"] for column in inspector.get_columns(table.name)}
        assert db_columns == {column.name for column in table.columns}
    with engine.connect() as conn:
        present = conn.execute(
            text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        ).scalar()
    engine.dispose()
    assert present is True


def test_fixed_records_round_trip(session: Session) -> None:
    session.add(
        Company(
            id=COMPANY_ID,
            ticker="ACME",
            cik="0000000001",
            name="Acme Industries",
            exchange="NYSE",
            sector="Industrials",
            industry="Machinery",
            market_cap=Decimal("2000000000"),
            created_at=CREATED,
        )
    )
    session.add(
        Document(
            id=DOCUMENT_ID,
            company_id=COMPANY_ID,
            document_type="8-K",
            accession_number="0000000001-22-000001",
            filing_date=date(2022, 5, 4),
            period_start=date(2022, 1, 1),
            period_end=date(2022, 3, 31),
            published_at=PUBLISHED,
            effective_at=PUBLISHED,
            retrieved_at=CREATED,
            source_url="https://www.sec.gov/example",
            content="Guidance reduced.",
            content_hash="hash-0001",
            created_at=CREATED,
        )
    )
    session.add(
        Observation(
            id=OBSERVATION_ID,
            company_id=COMPANY_ID,
            event_type="GUIDANCE_REVISION",
            title="Guidance cut",
            description="Forward revenue guidance fell.",
            evidence={"magnitude": -0.08, "source": "8-K"},
            confidence=Decimal("0.94"),
            detected_at=DETECTED,
            published_at=PUBLISHED,
            effective_at=PUBLISHED,
            period_start=date(2022, 1, 1),
            period_end=date(2022, 3, 31),
            retrieved_at=CREATED,
            created_at=CREATED,
        )
    )
    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="Guidance cuts predict weaker returns",
            statement="Negative guidance revisions predict negative sector-relative returns.",
            universe={"asset_class": "US_EQUITY", "min_market_cap": 2000000000},
            mechanism="Investors reprice expected cash flows.",
            entry_rule={"timing": "NEXT_MARKET_OPEN"},
            exit_rule={"holding_period_days": 60},
            expected_horizon=60,
            status=HypothesisStatus.DRAFT.value,
            created_at=CREATED,
        )
    )
    session.flush()
    session.add(
        Hypothesis(
            id=CHILD_HYPOTHESIS_ID,
            title="Follow-up",
            statement="The effect is stronger above five percent.",
            parent_hypothesis_id=HYPOTHESIS_ID,
            status=HypothesisStatus.UNDER_TEST.value,
            created_at=CREATED,
        )
    )
    session.flush()
    session.add(
        Experiment(
            id=EXPERIMENT_ID,
            hypothesis_id=HYPOTHESIS_ID,
            dataset_version="sec-2024-01",
            data_cutoff=datetime(2023, 12, 31, 23, 59, tzinfo=UTC),
            code_version="a1b2c3d",
            experiment_timestamp=datetime(2024, 3, 2, 12, 0, tzinfo=UTC),
            train_start=date(2016, 1, 1),
            train_end=date(2019, 12, 31),
            validation_start=date(2020, 1, 1),
            validation_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2023, 12, 31),
            specification={"benchmark": "SECTOR"},
            parameters={"threshold": -0.05},
            metrics={"sharpe": 0.4},
            statistical_tests={"t_stat": 1.2},
            status="RESULTS_READY",
            started_at=STARTED,
            created_at=CREATED,
        )
    )
    session.flush()
    session.add(
        Experiment(
            id=CHILD_EXPERIMENT_ID,
            hypothesis_id=HYPOTHESIS_ID,
            dataset_version="sec-2024-01",
            code_version="a1b2c3d",
            status="RUNNING",
            parent_experiment_id=EXPERIMENT_ID,
            created_at=CREATED,
        )
    )
    session.add(
        ResearchArtifact(
            id=ARTIFACT_ID,
            experiment_id=EXPERIMENT_ID,
            artifact_type=ArtifactType.BACKTEST_REPORT.value,
            content="Backtest report body.",
            model_name="none",
            model_version="0",
            prompt_version="0",
            created_at=CREATED,
        )
    )
    session.add(
        Strategy(
            id=STRATEGY_ID,
            experiment_id=EXPERIMENT_ID,
            name="Guidance revision",
            version=2,
            signal_definition={"event": "GUIDANCE_REVISION"},
            portfolio_rules={"weighting": "equal"},
            risk_rules={"max_position": 0.05},
            capacity_estimate=Decimal("1000000"),
            status="CANDIDATE",
            created_at=CREATED,
        )
    )
    session.add(
        Strategy(
            id=DEFAULT_STRATEGY_ID,
            experiment_id=EXPERIMENT_ID,
            name="Default version",
            created_at=CREATED,
        )
    )
    session.commit()
    try:
        _assert_round_trip(session)
    finally:
        _delete_round_trip(session)


def _assert_round_trip(session: Session) -> None:
    stored_company = session.get(Company, COMPANY_ID)
    stored_document = session.get(Document, DOCUMENT_ID)
    stored_observation = session.get(Observation, OBSERVATION_ID)
    stored_hypothesis = session.get(Hypothesis, HYPOTHESIS_ID)
    stored_experiment = session.get(Experiment, EXPERIMENT_ID)
    stored_artifact = session.get(ResearchArtifact, ARTIFACT_ID)
    stored_strategy = session.get(Strategy, STRATEGY_ID)
    assert stored_company is not None
    assert stored_document is not None
    assert stored_observation is not None
    assert stored_hypothesis is not None
    assert stored_experiment is not None
    assert stored_artifact is not None
    assert stored_strategy is not None
    assert CompanyRecord.model_validate(stored_company).market_cap == Decimal("2000000000")
    document_record = DocumentRecord.model_validate(stored_document)
    assert document_record.published_at == PUBLISHED
    assert document_record.effective_at == PUBLISHED
    assert document_record.period_start == date(2022, 1, 1)
    assert document_record.period_end == date(2022, 3, 31)
    assert document_record.retrieved_at == CREATED
    assert ObservationRecord.model_validate(stored_observation).evidence == {
        "magnitude": -0.08,
        "source": "8-K",
    }
    assert HypothesisRecord.model_validate(stored_hypothesis).status is HypothesisStatus.DRAFT
    assert ExperimentRecord.model_validate(stored_experiment).metrics == {"sharpe": 0.4}
    assert stored_experiment.dataset_version == "sec-2024-01"
    assert stored_experiment.data_cutoff == datetime(2023, 12, 31, 23, 59, tzinfo=UTC)
    assert stored_experiment.code_version == "a1b2c3d"
    assert stored_experiment.experiment_timestamp == datetime(2024, 3, 2, 12, 0, tzinfo=UTC)
    child = session.get(Experiment, CHILD_EXPERIMENT_ID)
    assert child is not None
    assert child.parent_experiment_id == EXPERIMENT_ID
    assert ResearchArtifactRecord.model_validate(stored_artifact).artifact_type is (
        ArtifactType.BACKTEST_REPORT
    )
    assert StrategyRecord.model_validate(stored_strategy).version == 2
    default_strategy = session.get(Strategy, DEFAULT_STRATEGY_ID)
    assert default_strategy is not None
    assert default_strategy.version == 1


def _delete_round_trip(session: Session) -> None:
    for model, row_id in (
        (Strategy, DEFAULT_STRATEGY_ID),
        (Strategy, STRATEGY_ID),
        (ResearchArtifact, ARTIFACT_ID),
        (Experiment, CHILD_EXPERIMENT_ID),
        (Experiment, EXPERIMENT_ID),
        (Hypothesis, CHILD_HYPOTHESIS_ID),
        (Hypothesis, HYPOTHESIS_ID),
        (Observation, OBSERVATION_ID),
        (Document, DOCUMENT_ID),
        (Company, COMPANY_ID),
    ):
        row = session.get(model, row_id)
        if row is not None:
            session.delete(row)
    session.commit()


def test_constraints_reject_invalid_rows(session: Session) -> None:
    session.add(Company(id=COMPANY_ID, ticker="ACME", cik="0000000001", name="Acme"))
    session.flush()
    session.add(
        Company(
            id=UUID("00000000-0000-4000-8000-000000000011"),
            ticker="ACME2",
            cik="0000000001",
            name="Other",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(Company(id=COMPANY_ID, ticker="ACME", cik="0000000001", name="Acme"))
    session.add(
        Document(
            id=DOCUMENT_ID,
            company_id=COMPANY_ID,
            document_type="10-K",
            content_hash="hash-0001",
        )
    )
    session.flush()
    session.add(
        Document(
            id=UUID("00000000-0000-4000-8000-000000000012"),
            company_id=COMPANY_ID,
            document_type="10-Q",
            content_hash="hash-0001",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="Bad",
            statement="Bad",
            status="NOT_A_STATUS",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        ResearchArtifact(
            id=ARTIFACT_ID,
            artifact_type="MEMO",
            content="no",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        Document(
            id=DOCUMENT_ID,
            company_id=UUID("00000000-0000-4000-8000-000000000099"),
            document_type="8-K",
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    with pytest.raises(IntegrityError):
        session.execute(
            text(
                "INSERT INTO experiments (id, dataset_version, code_version, status) "
                "VALUES (:id, 'd', 'c', 'RUNNING')"
            ),
            {"id": EXPERIMENT_ID},
        )
    session.rollback()

    session.add(Company(id=COMPANY_ID, ticker="ACME", name="Acme"))
    session.add(Document(id=DOCUMENT_ID, company_id=COMPANY_ID, document_type="8-K"))
    session.flush()
    session.delete(session.get(Company, COMPANY_ID))
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="Dates",
            statement="Dates",
            status=HypothesisStatus.DRAFT.value,
        )
    )
    session.flush()
    session.add(
        Experiment(
            id=EXPERIMENT_ID,
            hypothesis_id=HYPOTHESIS_ID,
            dataset_version="d",
            code_version="c",
            status="RUNNING",
            train_start=date(2021, 1, 2),
            train_end=date(2021, 1, 1),
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="Self",
            statement="Self",
            status=HypothesisStatus.DRAFT.value,
            parent_hypothesis_id=HYPOTHESIS_ID,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()

    session.add(
        Strategy(
            id=STRATEGY_ID,
            name="Bad version",
            version=0,
        )
    )
    with pytest.raises(IntegrityError):
        session.flush()
    session.rollback()


def test_completed_experiment_is_immutable(session: Session) -> None:
    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="Locked",
            statement="Locked",
            status=HypothesisStatus.FAILED.value,
        )
    )
    session.flush()
    experiment = Experiment(
        id=EXPERIMENT_ID,
        hypothesis_id=HYPOTHESIS_ID,
        dataset_version="sec-2024-01",
        code_version="a1b2c3d",
        status="COMPLETED",
        completed_at=COMPLETED,
    )
    session.add(experiment)
    session.flush()
    experiment.metrics = {"sharpe": 9}
    with pytest.raises(DBAPIError, match="completed experiments are immutable"):
        session.flush()
    session.rollback()

    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="Locked",
            statement="Locked",
            status=HypothesisStatus.FAILED.value,
        )
    )
    session.flush()
    session.add(
        Experiment(
            id=EXPERIMENT_ID,
            hypothesis_id=HYPOTHESIS_ID,
            dataset_version="sec-2024-01",
            code_version="a1b2c3d",
            status="COMPLETED",
            completed_at=COMPLETED,
        )
    )
    session.flush()
    session.delete(session.get(Experiment, EXPERIMENT_ID))
    with pytest.raises(DBAPIError, match="completed experiments are immutable"):
        session.flush()
    session.rollback()

    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="Open",
            statement="Open",
            status=HypothesisStatus.DRAFT.value,
        )
    )
    session.flush()
    open_experiment = Experiment(
        id=EXPERIMENT_ID,
        hypothesis_id=HYPOTHESIS_ID,
        dataset_version="sec-2024-01",
        code_version="a1b2c3d",
        status="RUNNING",
    )
    session.add(open_experiment)
    session.flush()
    open_experiment.completed_at = COMPLETED
    session.flush()
    open_experiment.status = "COMPLETED"
    with pytest.raises(DBAPIError, match="completed experiments are immutable"):
        session.flush()


def test_z_downgrade_and_upgrade_recreate_the_same_catalog(migrated_database: str) -> None:
    del migrated_database
    before = _catalog()
    _alembic("downgrade")
    tables, _, _, trigger = _catalog()
    assert RESEARCH_TABLES.isdisjoint(tables)
    assert trigger is None
    _alembic("upgrade")
    assert _catalog() == before
