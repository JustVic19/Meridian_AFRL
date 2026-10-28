"""Experiment integrity. Test results stay hidden until lock and are never rewritten."""

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from api.models import Experiment, Hypothesis
from research.backtest.engine import Bar, Signal
from research.experiments import (
    ExperimentEngine,
    ExperimentIntegrityError,
    ExperimentParameters,
    ExperimentState,
    Period,
)
from schemas.experiment_spec import DataRequirement, DatasetCatalog, ExperimentSpecification
from schemas.hypothesis import EntryRule, EventDefinition, ExitRule, UniverseDefinition
from schemas.research import HypothesisStatus

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_experiment_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"

HYPOTHESIS_ID = UUID("00000000-0000-4000-8000-0000000000f2")
REVISED_HYPOTHESIS_ID = UUID("00000000-0000-4000-8000-0000000000f3")
OPENED = datetime(2020, 1, 10, 21, tzinfo=UTC)
TESTED = datetime(2020, 1, 11, 21, tzinfo=UTC)
START = date(2020, 1, 1)


def _specification(code_version: str = "a1b2c3d") -> ExperimentSpecification:
    return ExperimentSpecification(
        universe=UniverseDefinition(asset_class="US_EQUITY", min_market_cap=2_000_000_000),
        event=EventDefinition(type="GUIDANCE_REVISION", threshold=-0.05),
        entry=EntryRule(timing="NEXT_MARKET_OPEN"),
        exit=ExitRule(holding_period_days=1),
        benchmark="SPX",
        data_requirements=[DataRequirement(name="prices", fields=["close"])],
        required_fields=["close"],
        transaction_cost_bps=0,
        slippage_bps=0,
        random_seed=7,
        dataset_version="prices-2020-01",
        code_version=code_version,
        data_cutoff=OPENED,
    )


def _catalog() -> DatasetCatalog:
    return DatasetCatalog(
        dataset_version="prices-2020-01",
        start=START,
        end=date(2020, 1, 10),
        fields=["close"],
        published_at=datetime(2020, 1, 10, 18, tzinfo=UTC),
    )


def _parameters() -> ExperimentParameters:
    return ExperimentParameters(
        initial_cash=Decimal(100),
        entry_timing="CLOSE",
        holding_period_sessions=1,
        sizing_fraction=Decimal(1),
        transaction_cost_bps=Decimal(0),
        slippage_bps=Decimal(0),
        periods_per_year=1,
        benchmark_symbol="SPX",
        signals=(
            Signal("AAA", date(2020, 1, 1)),
            Signal("AAA", date(2020, 1, 7)),
            Signal("AAA", date(2020, 1, 9)),
        ),
    )


def _bars() -> tuple[Bar, ...]:
    rows: list[Bar] = []
    for offset in range(10):
        session = START + timedelta(days=offset)
        price = Decimal(200) if offset == 9 else Decimal(100)
        rows.append(Bar("AAA", session, price))
        rows.append(Bar("SPX", session, Decimal(100)))
    return tuple(rows)


def _store_hypotheses(session: Session) -> None:
    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="Guidance",
            statement="A precise hypothesis with a 5 percent threshold over 60 days.",
            status=HypothesisStatus.DRAFT.value,
        )
    )
    session.add(
        Hypothesis(
            id=REVISED_HYPOTHESIS_ID,
            title="Guidance revised",
            statement="A precise hypothesis with an 8 percent threshold over 60 days.",
            status=HypothesisStatus.DRAFT.value,
        )
    )
    session.flush()


def _prepare(engine: ExperimentEngine) -> UUID:
    created = engine.create(
        hypothesis_id=HYPOTHESIS_ID,
        specification=_specification(),
        catalog=_catalog(),
        parameters=_parameters(),
        executed_at=OPENED,
    )
    bars = _bars()
    engine.record_train(created.experiment_id, bars)
    engine.record_validation(created.experiment_id, bars)
    engine.lock(created.experiment_id)
    return created.experiment_id


def test_experiment_creation_uses_the_default_split(session: Session) -> None:
    _store_hypotheses(session)
    engine = ExperimentEngine(session)
    first = engine.create(
        hypothesis_id=HYPOTHESIS_ID,
        specification=_specification(),
        catalog=_catalog(),
        parameters=_parameters(),
        executed_at=OPENED,
    )
    second = engine.create(
        hypothesis_id=HYPOTHESIS_ID,
        specification=_specification(),
        catalog=_catalog(),
        parameters=_parameters(),
        executed_at=datetime(2020, 2, 1, tzinfo=UTC),
    )
    assert first.experiment_id == second.experiment_id
    assert first.status is ExperimentState.CREATED
    assert first.hypothesis_id == HYPOTHESIS_ID
    assert first.dataset_version == "prices-2020-01"
    assert first.code_version == "a1b2c3d"
    assert first.train_start == date(2020, 1, 1)
    assert first.train_end == date(2020, 1, 6)
    assert first.validation_start == date(2020, 1, 7)
    assert first.validation_end == date(2020, 1, 8)
    assert first.test_start == date(2020, 1, 9)
    assert first.test_end == date(2020, 1, 10)
    assert first.parameters["initial_cash"] == "100"
    assert first.started_at == OPENED
    assert first.experiment_timestamp == OPENED
    assert first.completed_at is None
    assert first.parent_experiment_id is None
    assert "test" not in first.metrics
    count = session.scalar(select(func.count()).select_from(Experiment))
    assert count == 1


def test_periods_are_isolated(session: Session) -> None:
    _store_hypotheses(session)
    engine = ExperimentEngine(session)
    created = engine.create(
        hypothesis_id=HYPOTHESIS_ID,
        specification=_specification(),
        catalog=_catalog(),
        parameters=_parameters(),
        executed_at=OPENED,
    )
    bars = _bars()
    with pytest.raises(ExperimentIntegrityError, match="test period is sealed"):
        engine.bars_for(created.experiment_id, Period.TEST, bars)
    train = engine.record_train(created.experiment_id, bars)
    validation = engine.record_validation(created.experiment_id, bars)
    assert train.metrics["train"]["number_of_observations"] == 5
    assert validation.metrics["validation"]["number_of_observations"] == 1
    assert train.metrics["train"]["equity_end"] == "100"
    assert "200" not in str(train.metrics)
    assert "test" not in validation.metrics
    assert "test" not in validation.statistical_tests


def test_test_results_stay_sealed_until_lock(session: Session) -> None:
    _store_hypotheses(session)
    engine = ExperimentEngine(session)
    created = engine.create(
        hypothesis_id=HYPOTHESIS_ID,
        specification=_specification(),
        catalog=_catalog(),
        parameters=_parameters(),
        executed_at=OPENED,
    )
    bars = _bars()
    engine.record_train(created.experiment_id, bars)
    engine.record_validation(created.experiment_id, bars)
    with pytest.raises(ExperimentIntegrityError, match="does not allow"):
        engine.record_test(created.experiment_id, bars, executed_at=TESTED)
    locked = engine.lock(created.experiment_id)
    assert locked.status is ExperimentState.LOCKED
    assert "test" not in locked.metrics
    revealed = engine.record_test(created.experiment_id, bars, executed_at=TESTED)
    assert revealed.status is ExperimentState.TEST_RECORDED
    assert revealed.metrics["test"]["number_of_observations"] == 1
    assert revealed.metrics["test"]["equity_end"] == "200"
    assert revealed.completed_at == TESTED


def test_recorded_test_results_are_immutable(session: Session) -> None:
    _store_hypotheses(session)
    engine = ExperimentEngine(session)
    experiment_id = _prepare(engine)
    bars = _bars()
    recorded = engine.record_test(experiment_id, bars, executed_at=TESTED)
    replayed = engine.record_test(
        experiment_id,
        bars,
        executed_at=datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert replayed.metrics == recorded.metrics
    assert replayed.completed_at == TESTED
    stored = session.get(Experiment, experiment_id)
    assert stored is not None
    stored.metrics = {"test": {"equity_end": "1"}}
    with pytest.raises(DBAPIError, match="completed experiments are immutable"):
        session.flush()


def test_a_changed_design_branches_without_overwriting_the_parent(session: Session) -> None:
    _store_hypotheses(session)
    engine = ExperimentEngine(session)
    created = engine.create(
        hypothesis_id=HYPOTHESIS_ID,
        specification=_specification(),
        catalog=_catalog(),
        parameters=_parameters(),
        executed_at=OPENED,
    )
    with pytest.raises(ExperimentIntegrityError, match="have not been recorded"):
        engine.revise(
            created.experiment_id,
            hypothesis_id=REVISED_HYPOTHESIS_ID,
            specification=_specification("revised-1"),
            catalog=_catalog(),
            parameters=_parameters(),
            executed_at=TESTED,
        )
    experiment_id = _prepare(engine)
    engine.record_test(experiment_id, _bars(), executed_at=TESTED)
    parent_before = session.get(Experiment, experiment_id)
    assert parent_before is not None
    parent_metrics = dict(parent_before.metrics or {})
    child = engine.revise(
        experiment_id,
        hypothesis_id=REVISED_HYPOTHESIS_ID,
        specification=_specification("revised-1"),
        catalog=_catalog(),
        parameters=_parameters(),
        executed_at=TESTED,
    )
    parent_after = session.get(Experiment, experiment_id)
    assert parent_after is not None
    assert parent_after.metrics == parent_metrics
    assert parent_after.hypothesis_id == HYPOTHESIS_ID
    assert parent_after.code_version == "a1b2c3d"
    assert parent_after.status == ExperimentState.TEST_RECORDED.value
    assert child.experiment_id != experiment_id
    assert child.parent_experiment_id == experiment_id
    assert child.hypothesis_id == REVISED_HYPOTHESIS_ID
    assert child.code_version == "revised-1"
    assert child.status is ExperimentState.CREATED
    assert "test" not in child.metrics
    assert engine.lineage(child.experiment_id) == (experiment_id, child.experiment_id)
    assert engine.lineage(experiment_id) == (experiment_id,)
    again = engine.revise(
        experiment_id,
        hypothesis_id=REVISED_HYPOTHESIS_ID,
        specification=_specification("revised-1"),
        catalog=_catalog(),
        parameters=_parameters(),
        executed_at=datetime(2022, 1, 1, tzinfo=UTC),
    )
    assert again.experiment_id == child.experiment_id
    count = session.scalar(select(func.count()).select_from(Experiment))
    assert count == 2


def test_replay_repeats_the_same_metrics(session: Session) -> None:
    _store_hypotheses(session)
    engine = ExperimentEngine(session)
    experiment_id = _prepare(engine)
    bars = _bars()
    first = engine.record_test(experiment_id, bars, executed_at=TESTED)
    second = engine.record_test(experiment_id, bars, executed_at=TESTED)
    assert first == second
    assert first.statistical_tests["test"]["observations"] == 1


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(sql))
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_experiment_database() -> Iterator[None]:
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
def session(migrated_experiment_database: None) -> Iterator[Session]:
    del migrated_experiment_database
    engine = create_engine(TEST_URL)
    with Session(engine) as db:
        try:
            yield db
        finally:
            db.rollback()
    engine.dispose()
