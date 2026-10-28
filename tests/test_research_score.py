"""Research score. Weights are fixed, and the score does not authorize a trade."""

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime
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
from api.models import ResearchScore as StoredScore
from research.scoring import (
    ResearchScoreEngine,
    Scorecard,
    ScoreRejected,
    evaluate,
    policy_for,
)
from schemas.research import HypothesisStatus
from schemas.scoring import ComponentStatus, ScoreComponent, ScoreStatus
from schemas.skeptic import CheckGrade

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_score_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"

EXPERIMENT_ID = UUID("00000000-0000-4000-8000-0000000002a1")
HYPOTHESIS_ID = UUID("00000000-0000-4000-8000-0000000002b1")
EXECUTED = datetime(2020, 1, 13, tzinfo=UTC)
WEIGHTS = (
    Decimal("0.20"),
    Decimal("0.15"),
    Decimal("0.15"),
    Decimal("0.15"),
    Decimal("0.10"),
    Decimal("0.10"),
    Decimal("0.10"),
    Decimal("0.05"),
)


def test_weights_are_applied_exactly() -> None:
    policy = policy_for("research-score-1")
    assert [weight for _component, weight in policy.weights] == list(WEIGHTS)
    assert sum(WEIGHTS, Decimal(0)) == Decimal("1.00")
    assert evaluate(_card(), policy).score == Decimal(1)
    only_oos = evaluate(
        _card(
            robustness_completed=0,
            test_observations=0,
            mechanism=CheckGrade.FAIL,
            stressed_cost_sharpe=Decimal(0),
            regime_sharpes=(Decimal(1), Decimal(0)),
            capacity=CheckGrade.FAIL,
            specifications_searched=10,
        ),
        policy,
    )
    assert only_oos.score == Decimal("0.20")
    mixed = evaluate(
        _card(
            oos_sharpe=Decimal("0.5"),
            robustness_completed=2,
            test_observations=65,
            mechanism=CheckGrade.CONDITIONAL,
            stressed_cost_sharpe=Decimal("0.5"),
            regime_sharpes=(Decimal(1), Decimal("0.5")),
            capacity=CheckGrade.CONDITIONAL,
        ),
        policy,
    )
    assert mixed.score == Decimal("0.525")
    assert mixed.status is ScoreStatus.SCORED


def test_boundaries_stay_inside_the_unit_interval() -> None:
    policy = policy_for("research-score-1")
    assert _score(_card(oos_sharpe=Decimal(-1)), ScoreComponent.OOS_PERFORMANCE) == Decimal(0)
    assert _score(_card(oos_sharpe=Decimal(0)), ScoreComponent.OOS_PERFORMANCE) == Decimal(0)
    assert _score(_card(oos_sharpe=Decimal(3)), ScoreComponent.OOS_PERFORMANCE) == Decimal(1)
    assert _score(_card(test_observations=29), ScoreComponent.STATISTICAL_CONFIDENCE) == Decimal(0)
    assert _score(_card(test_observations=30), ScoreComponent.STATISTICAL_CONFIDENCE) == Decimal(0)
    assert _score(_card(test_observations=100), ScoreComponent.STATISTICAL_CONFIDENCE) == Decimal(1)
    assert _score(_card(test_observations=101), ScoreComponent.STATISTICAL_CONFIDENCE) == Decimal(1)
    assert _score(_card(robustness_completed=0), ScoreComponent.ROBUSTNESS) == Decimal(0)
    assert _score(_card(robustness_completed=4), ScoreComponent.ROBUSTNESS) == Decimal(1)
    assert _score(_card(specifications_searched=1), ScoreComponent.SIMPLICITY) == Decimal(1)
    assert _score(_card(specifications_searched=10), ScoreComponent.SIMPLICITY) == Decimal(0)
    assert _score(_card(specifications_searched=11), ScoreComponent.SIMPLICITY) == Decimal(0)
    spread = _card(regime_sharpes=(Decimal(1), Decimal(2)))
    assert _score(spread, ScoreComponent.REGIME_STABILITY) == Decimal(0)
    failed = _card(transaction_cost_failed=True, baseline_sharpe=None, stressed_cost_sharpe=None)
    assert _score(failed, ScoreComponent.TRANSACTION_COST_RESILIENCE) == Decimal(0)
    assert evaluate(failed, policy).score == Decimal("0.90")
    with pytest.raises(ScoreRejected):
        evaluate(_card(robustness_completed=5), policy)


def test_missing_metrics_withhold_the_total() -> None:
    policy = policy_for("research-score-1")
    missing = evaluate(_card(oos_sharpe=None), policy)
    assert missing.status is ScoreStatus.INCOMPLETE
    assert missing.score is None
    oos = _component(missing, ScoreComponent.OOS_PERFORMANCE)
    assert oos.status is ComponentStatus.MISSING
    assert oos.score is None
    assert oos.weight == Decimal("0.20")
    others = [item for item in missing.components if item.component is not oos.component]
    assert {item.status for item in others} == {ComponentStatus.SCORED}
    assert {item.score for item in others} == {Decimal(1)}
    one_regime = evaluate(_card(regime_sharpes=(Decimal(1),)), policy)
    assert one_regime.score is None
    assert _component(one_regime, ScoreComponent.REGIME_STABILITY).status is ComponentStatus.MISSING


def test_the_same_card_replays_one_score(session: Session) -> None:
    _store_experiment(session)
    engine = ResearchScoreEngine(session)
    first = engine.score(EXPERIMENT_ID, _card(), executed_at=EXECUTED)
    second = engine.score(
        EXPERIMENT_ID,
        _card(),
        executed_at=datetime(2021, 1, 1, tzinfo=UTC),
    )
    assert first.id == second.id
    assert first.score == second.score == Decimal(1)
    assert first.executed_at == EXECUTED
    assert first.live_trade_authorized is False
    assert "does not authorize a live trade" in first.disclaimer
    with pytest.raises(ScoreRejected, match="immutable"):
        engine.score(EXPERIMENT_ID, _card(oos_sharpe=Decimal(0)), executed_at=EXECUTED)
    stored = session.get(StoredScore, first.id)
    assert stored is not None
    stored.score = "0.00000000"
    with pytest.raises(DBAPIError, match="research scores are immutable"):
        session.flush()


def test_policy_version_is_stored_with_the_weights(session: Session) -> None:
    _store_experiment(session, status="LOCKED", completed_at=None)
    engine = ResearchScoreEngine(session)
    with pytest.raises(ScoreRejected, match="not recorded"):
        engine.score(EXPERIMENT_ID, _card(), executed_at=EXECUTED)
    with pytest.raises(ScoreRejected, match="not registered"):
        policy_for("research-score-tuned")
    assert _count(session) == 0
    session.rollback()
    _store_experiment(session)
    stored_score = engine.score(
        EXPERIMENT_ID,
        _card(),
        executed_at=EXECUTED,
        policy_version="research-score-1",
    )
    stored = session.get(StoredScore, stored_score.id)
    assert stored is not None
    assert stored.policy_version == "research-score-1"
    assert stored.live_trade_authorized is False
    assert stored.weights == {
        "oos_performance": "0.20",
        "robustness": "0.15",
        "statistical_confidence": "0.15",
        "economic_mechanism": "0.15",
        "transaction_cost_resilience": "0.10",
        "regime_stability": "0.10",
        "capacity": "0.10",
        "simplicity": "0.05",
    }
    assert stored.policy["version"] == "research-score-1"
    assert stored.policy["purpose"] == "research_ranking"
    assert stored.policy["live_trade_authorized"] is False
    assert stored.score == "1.00000000"
    assert _count(session) == 1


def _card(**overrides: object) -> Scorecard:
    values: dict[str, object] = {
        "oos_sharpe": Decimal(1),
        "robustness_completed": 4,
        "test_observations": 100,
        "mechanism": CheckGrade.PASS,
        "baseline_sharpe": Decimal(1),
        "stressed_cost_sharpe": Decimal(1),
        "regime_sharpes": (Decimal(1), Decimal(1)),
        "capacity": CheckGrade.PASS,
        "specifications_searched": 1,
    }
    values.update(overrides)
    return Scorecard(**values)  # type: ignore[arg-type]


def _component(evaluation: object, component: ScoreComponent) -> object:
    found = next(
        item
        for item in evaluation.components  # type: ignore[attr-defined]
        if item.component is component
    )
    return found


def _score(card: Scorecard, component: ScoreComponent) -> Decimal | None:
    found = _component(evaluate(card, policy_for("research-score-1")), component)
    return found.score  # type: ignore[attr-defined]


def _store_experiment(
    session: Session,
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
            id=EXPERIMENT_ID,
            hypothesis_id=HYPOTHESIS_ID,
            dataset_version="prices-2020-01",
            code_version="score-test",
            data_cutoff=EXECUTED,
            experiment_timestamp=EXECUTED,
            started_at=EXECUTED,
            completed_at=completed_at,
            train_start=date(2020, 1, 1),
            train_end=date(2020, 1, 4),
            validation_start=date(2020, 1, 5),
            validation_end=date(2020, 1, 6),
            test_start=date(2020, 1, 7),
            test_end=date(2020, 1, 12),
            specification={},
            parameters={},
            metrics={},
            statistical_tests={},
            status=status,
        )
    )
    session.flush()


def _count(session: Session) -> int:
    count = session.scalar(select(func.count()).select_from(StoredScore))
    assert count is not None
    return count


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(sql))
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_score_database() -> Iterator[None]:
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
def session(migrated_score_database: None) -> Iterator[Session]:
    del migrated_score_database
    engine = create_engine(TEST_URL)
    with Session(engine) as db:
        try:
            yield db
        finally:
            db.rollback()
    engine.dispose()
