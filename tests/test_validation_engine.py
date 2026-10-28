"""Validation engine. Every registered test runs, and a stored result is never rewritten."""

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
from api.models import ValidationResult as StoredResult
from api.models import ValidationRun as StoredRun
from financial.precision import q
from research.backtest.engine import Bar
from research.validation import (
    CURRENT_POLICY_VERSION,
    Observation,
    RegimeLabel,
    ValidationBook,
    ValidationEngine,
    ValidationRejected,
    policy_for,
)
from schemas.research import HypothesisStatus
from schemas.validation import ValidationStatus, ValidationTest

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_validation_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"

LEFT = UUID("00000000-0000-4000-8000-0000000001a1")
RIGHT = UUID("00000000-0000-4000-8000-0000000001a2")
BROKEN = UUID("00000000-0000-4000-8000-0000000001a3")
GAP = UUID("00000000-0000-4000-8000-0000000001a4")
HYPOTHESIS_ID = UUID("00000000-0000-4000-8000-0000000001b1")
EXECUTED = datetime(2020, 1, 13, tzinfo=UTC)
START = date(2020, 1, 1)
VALIDATION_END = date(2020, 1, 6)
TEST_START = date(2020, 1, 7)
TEST_END = date(2020, 1, 12)
ZERO = str(q(Decimal(0)))


def test_every_required_test_runs_in_policy_order(session: Session) -> None:
    _store_experiment(session, LEFT)
    engine = ValidationEngine(session)
    run = engine.run(LEFT, _book(), executed_at=EXECUTED)
    assert run.policy.version == CURRENT_POLICY_VERSION
    assert run.policy.sequence == tuple(ValidationTest)
    assert [result.test for result in run.results] == list(ValidationTest)
    assert [result.sequence_index for result in run.results] == list(range(8))
    assert {result.status for result in run.results} == {ValidationStatus.COMPLETE}
    assert run.status is ValidationStatus.COMPLETE
    assert all(result.detail["executed"] is True for result in run.results)
    by_name = {result.test: result for result in run.results}
    assert by_name[ValidationTest.BASELINE].detail["window"] == ["2020-01-01", "2020-01-06"]
    assert by_name[ValidationTest.BASELINE].detail["transaction_cost_bps"] == "0"
    assert by_name[ValidationTest.TRANSACTION_COST].detail["transaction_cost_bps"] == "10"
    assert by_name[ValidationTest.OUT_OF_SAMPLE].detail["window"] == ["2020-01-07", "2020-01-12"]
    assert by_name[ValidationTest.ALTERNATIVE_THRESHOLD].detail["threshold"] == "6"
    assert by_name[ValidationTest.ALTERNATIVE_HOLDING_PERIOD].detail["holding_period_sessions"] == 2
    assert _part_names(by_name[ValidationTest.SUBPERIOD]) == ["first", "second"]
    assert _part_names(by_name[ValidationTest.REGIME]) == ["calm", "shock"]
    assert _part_names(by_name[ValidationTest.SECTOR]) == ["FIN", "TECH"]
    metrics = by_name[ValidationTest.BASELINE].metrics
    assert metrics is not None
    assert metrics["number_of_observations"] > 0
    assert "not indicative of future results" in str(metrics["disclaimer"])
    assert _artifact_count(session) == 8


def test_a_failed_test_remains_in_the_sequence(session: Session) -> None:
    _store_experiment(session, LEFT)
    _store_experiment(session, BROKEN)
    _store_experiment(session, GAP)
    engine = ValidationEngine(session)
    engine.run(LEFT, _book(), executed_at=EXECUTED)
    broken = engine.run(BROKEN, _book(start=TEST_START), executed_at=EXECUTED)
    assert [result.test for result in broken.results] == list(ValidationTest)
    by_name = {result.test: result for result in broken.results}
    assert by_name[ValidationTest.BASELINE].status is ValidationStatus.FAILED
    assert by_name[ValidationTest.TRANSACTION_COST].status is ValidationStatus.FAILED
    assert by_name[ValidationTest.OUT_OF_SAMPLE].status is ValidationStatus.COMPLETE
    assert by_name[ValidationTest.SECTOR].status is ValidationStatus.COMPLETE
    assert broken.status is ValidationStatus.FAILED
    assert "no tradable sessions" in str(by_name[ValidationTest.BASELINE].detail["error"])
    missing = engine.run(GAP, _book(sector=""), executed_at=EXECUTED)
    sector = {result.test: result for result in missing.results}[ValidationTest.SECTOR]
    assert sector.status is ValidationStatus.FAILED
    assert "sector" in str(sector.detail["error"])
    compared = engine.compare(LEFT, BROKEN)
    assert [row.test for row in compared.tests] == list(ValidationTest)
    assert compared.tests[0].left_status is ValidationStatus.COMPLETE
    assert compared.tests[0].right_status is ValidationStatus.FAILED


def test_recorded_results_are_immutable(session: Session) -> None:
    _store_experiment(session, LEFT)
    engine = ValidationEngine(session)
    recorded = engine.run(LEFT, _book(), executed_at=EXECUTED)
    replayed = engine.run(LEFT, _book(), executed_at=datetime(2021, 1, 1, tzinfo=UTC))
    assert replayed.id == recorded.id
    assert replayed.executed_at == EXECUTED
    assert [result.id for result in replayed.results] == [result.id for result in recorded.results]
    with pytest.raises(ValidationRejected, match="immutable"):
        engine.run(LEFT, _book(flat=True), executed_at=EXECUTED)
    stored = session.scalars(select(StoredResult).where(StoredResult.run_id == recorded.id)).first()
    assert stored is not None
    original = stored.metrics
    stored.metrics = {"equity_end": "1"}
    with pytest.raises(DBAPIError, match="validation results are immutable"):
        session.flush()
    assert original != {"equity_end": "1"}


def test_policy_version_is_preserved(session: Session) -> None:
    _store_experiment(session, LEFT, status="LOCKED", completed_at=None)
    engine = ValidationEngine(session)
    with pytest.raises(ValidationRejected, match="not recorded"):
        engine.run(LEFT, _book(), executed_at=EXECUTED)
    with pytest.raises(ValidationRejected, match="not registered"):
        policy_for("validation-chosen-by-the-model")
    assert _run_count(session) == 0
    session.rollback()
    _store_experiment(session, LEFT)
    run = engine.run(LEFT, _book(), executed_at=EXECUTED, policy_version=CURRENT_POLICY_VERSION)
    stored = session.get(StoredRun, run.id)
    assert stored is not None
    assert stored.policy_version == "validation-1"
    assert stored.policy["sequence"] == [test.value for test in ValidationTest]
    assert stored.policy["cost_stress_bps"] == "10"
    assert stored.policy["threshold_shift"] == "1"
    assert stored.policy["holding_shift"] == 1
    assert run.policy == policy_for("validation-1")
    assert _run_count(session) == 1


def test_experiments_can_be_compared(session: Session) -> None:
    _store_experiment(session, LEFT)
    _store_experiment(session, RIGHT)
    engine = ValidationEngine(session)
    engine.run(LEFT, _book(), executed_at=EXECUTED)
    engine.run(RIGHT, _book(flat=True), executed_at=EXECUTED)
    same = engine.compare(LEFT, LEFT)
    different = engine.compare(LEFT, RIGHT)
    assert same.policy_version == "validation-1"
    assert [row.test for row in same.tests] == list(ValidationTest)
    assert same.tests[0].equity_end_delta == ZERO
    assert different.left_experiment_id == LEFT
    assert different.right_experiment_id == RIGHT
    assert different.tests[0].equity_end_delta != ZERO
    assert different.tests[0].left_equity_end != different.tests[0].right_equity_end
    with pytest.raises(ValidationRejected, match="does not exist"):
        engine.compare(LEFT, BROKEN)


def _part_names(result: object) -> list[object]:
    parts = result.detail["parts"]  # type: ignore[attr-defined]
    assert isinstance(parts, list)
    return [part["name"] for part in parts]


def _book(
    *,
    flat: bool = False,
    start: date = START,
    sector: str = "FIN",
) -> ValidationBook:
    return ValidationBook(
        bars=_bars(flat=flat, start=start),
        observations=(
            Observation("AAA", date(2020, 1, 2), Decimal(5), "TECH"),
            Observation("AAA", date(2020, 1, 8), Decimal(5), "TECH"),
            Observation("BBB", date(2020, 1, 9), Decimal(8), sector),
        ),
        regimes=(
            RegimeLabel("shock", date(2020, 1, 10), date(2020, 1, 12)),
            RegimeLabel("calm", date(2020, 1, 7), date(2020, 1, 9)),
        ),
    )


def _bars(*, flat: bool, start: date) -> tuple[Bar, ...]:
    rows: list[Bar] = []
    day = start
    while day <= TEST_END:
        offset = (day - START).days
        price = Decimal(100) if flat else Decimal(100 + offset)
        rows.append(Bar("AAA", day, price))
        rows.append(Bar("BBB", day, Decimal(100)))
        rows.append(Bar("SPX", day, Decimal(100)))
        day += timedelta(days=1)
    return tuple(rows)


def _store_experiment(
    session: Session,
    experiment_id: UUID,
    *,
    status: str = "TEST_RECORDED",
    completed_at: datetime | None = EXECUTED,
) -> None:
    if session.get(Hypothesis, HYPOTHESIS_ID) is None:
        session.add(
            Hypothesis(
                id=HYPOTHESIS_ID,
                title="guidance_cut 5d",
                statement="A guidance cut of 5 percent precedes a 5 day return.",
                status=HypothesisStatus.DRAFT.value,
            )
        )
        session.flush()
    session.add(
        Experiment(
            id=experiment_id,
            hypothesis_id=HYPOTHESIS_ID,
            dataset_version="prices-2020-01",
            code_version="validation-test",
            data_cutoff=EXECUTED,
            experiment_timestamp=EXECUTED,
            started_at=EXECUTED,
            completed_at=completed_at,
            train_start=START,
            train_end=date(2020, 1, 4),
            validation_start=date(2020, 1, 5),
            validation_end=VALIDATION_END,
            test_start=TEST_START,
            test_end=TEST_END,
            specification={
                "specification": {
                    "benchmark": "SPX",
                    "event": {"type": "guidance_cut", "threshold": 5},
                }
            },
            parameters={
                "initial_cash": "10000",
                "entry_timing": "CLOSE",
                "holding_period_sessions": 1,
                "sizing_fraction": "1",
                "transaction_cost_bps": "0",
                "slippage_bps": "0",
                "periods_per_year": 1,
                "benchmark_symbol": "SPX",
                "risk_free_rate_per_period": "0",
            },
            metrics={},
            statistical_tests={},
            status=status,
        )
    )
    session.flush()


def _artifact_count(session: Session) -> int:
    count = session.scalar(select(func.count()).select_from(StoredResult))
    assert count is not None
    return count


def _run_count(session: Session) -> int:
    count = session.scalar(select(func.count()).select_from(StoredRun))
    assert count is not None
    return count


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(sql))
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_validation_database() -> Iterator[None]:
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
def session(migrated_validation_database: None) -> Iterator[Session]:
    del migrated_validation_database
    engine = create_engine(TEST_URL)
    with Session(engine) as db:
        try:
            yield db
        finally:
            db.rollback()
    engine.dispose()
