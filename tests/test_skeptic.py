"""Skeptic tests. Grades come from the experiment record, and the model is a script."""

import json
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
from sqlalchemy.orm import Session

from agents.llm import MockLanguageModel, ModelProviderError
from agents.skeptic.agent import (
    HypothesisDefense,
    MalformedModelResponse,
    SkepticAgent,
    SkepticRejected,
    UnsupportedEvidence,
)
from agents.skeptic.checks import (
    OVERFIT_FAIL_AT,
    SAMPLE_FAIL_BELOW,
    SAMPLE_PASS_AT,
    ExperimentSnapshot,
    grade_checks,
    verdict_for,
)
from api.models import Experiment, Hypothesis, ResearchArtifact
from prompts.skeptic import PROMPT_VERSION
from schemas.skeptic import AuditFacts, CheckGrade, CheckName, RegimeWindow, UniverseMember, Verdict

ROOT = Path(__file__).resolve().parents[1]
ADMIN_URL = "postgresql+psycopg://afil:afil_local_dev@localhost:5433/postgres"
TEST_DB = "afil_skeptic_test"
TEST_URL = f"postgresql+psycopg://afil:afil_local_dev@localhost:5433/{TEST_DB}"

EXPERIMENT_ID = UUID("00000000-0000-4000-8000-0000000000e1")
HYPOTHESIS_ID = UUID("00000000-0000-4000-8000-0000000000e2")
DECISION = datetime(2020, 1, 10, 21, tzinfo=UTC)
CUTOFF = datetime(2020, 1, 10, 21, tzinfo=UTC)
TRAIN_START = date(2020, 1, 1)
TRAIN_END = date(2020, 1, 6)
TEST_START = date(2020, 1, 9)
TEST_END = date(2020, 1, 10)
STATEMENT = "A guidance cut of 8 percent precedes weaker 5 day returns."
MECHANISM = "A guidance_cut changes expected cash flows, and prices adjust over the holding period."


def _audit(**overrides: object) -> AuditFacts:
    body: dict[str, object] = {
        "members": [UniverseMember(symbol="AAA", listed_on=date(2010, 1, 1))],
        "retained_delisted": True,
        "eligible_symbols": ["AAA"],
        "selected_symbols": ["AAA"],
        "selection_rule": "Every name in the point-in-time universe is eligible.",
        "feature_timestamps": [datetime(2020, 1, 10, 18, tzinfo=UTC)],
        "decision_timestamp": DECISION,
        "normalization_window": "train",
        "comparisons_tested": 1,
        "multiple_testing_adjustment": "none",
        "specifications_searched": 1,
        "regimes": [
            RegimeWindow(name="expansion", start=TRAIN_START, end=TEST_END),
            RegimeWindow(name="stress", start=TRAIN_START, end=TEST_END),
        ],
        "average_dollar_volume": {"AAA": Decimal("1000000")},
        "order_notional": Decimal("1000"),
        "participation_limit": Decimal("0.01"),
        "capacity_notional": Decimal("10000"),
        "benchmark_candidates": ["SPX", "IWM"],
    }
    body.update(overrides)
    return AuditFacts.model_validate(body)


def _snapshot(**overrides: object) -> ExperimentSnapshot:
    specification = {
        "specification": {
            "benchmark": "SPX",
            "transaction_cost_bps": 5,
            "slippage_bps": 5,
            "event": {"type": "guidance_cut", "threshold": 8},
        }
    }
    parameters = {"transaction_cost_bps": "5", "slippage_bps": "5", "benchmark_symbol": "SPX"}
    tests = {"test": {"observations": SAMPLE_PASS_AT, "mean_return": "0.01000000"}}
    body: dict[str, object] = {
        "data_cutoff": CUTOFF,
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "test_start": TEST_START,
        "test_end": TEST_END,
        "specification": specification,
        "parameters": parameters,
        "statistical_tests": tests,
        "hypothesis_statement": STATEMENT,
        "hypothesis_mechanism": MECHANISM,
    }
    body.update(overrides)
    return ExperimentSnapshot.model_validate(body)


def _grades(snapshot: ExperimentSnapshot | None = None, audit: AuditFacts | None = None) -> dict:
    findings = grade_checks(EXPERIMENT_ID, snapshot or _snapshot(), audit or _audit())
    return {item.check: item for item in findings}


def test_a_clean_record_passes_every_check() -> None:
    findings = grade_checks(EXPERIMENT_ID, _snapshot(), _audit())
    assert [item.check for item in findings] == list(CheckName)
    assert {item.grade for item in findings} == {CheckGrade.PASS}
    assert verdict_for([item.grade for item in findings]) is Verdict.PASS
    assert all(item.evidence for item in findings)


@pytest.mark.parametrize(
    ("check", "snapshot_changes", "audit_changes", "grade"),
    [
        (
            CheckName.LOOK_AHEAD_BIAS,
            {"data_cutoff": datetime(2020, 1, 9, 21, tzinfo=UTC)},
            {},
            CheckGrade.FAIL,
        ),
        (CheckName.SURVIVORSHIP_BIAS, {}, {"retained_delisted": False}, CheckGrade.FAIL),
        (CheckName.SELECTION_BIAS, {}, {"selected_symbols": ["BBB"]}, CheckGrade.FAIL),
        (CheckName.DATA_LEAKAGE, {}, {"normalization_window": "full_sample"}, CheckGrade.FAIL),
        (
            CheckName.MULTIPLE_TESTING,
            {},
            {"comparisons_tested": 4, "multiple_testing_adjustment": "none"},
            CheckGrade.FAIL,
        ),
        (CheckName.OVERFITTING, {}, {"specifications_searched": OVERFIT_FAIL_AT}, CheckGrade.FAIL),
        (
            CheckName.SAMPLE_SIZE,
            {"statistical_tests": {"test": {"observations": SAMPLE_FAIL_BELOW - 1}}},
            {},
            CheckGrade.FAIL,
        ),
        (
            CheckName.REGIME_DEPENDENCE,
            {},
            {"regimes": [RegimeWindow(name="stress", start=TEST_START, end=TEST_END)]},
            CheckGrade.FAIL,
        ),
        (
            CheckName.TRANSACTION_COSTS,
            {
                "parameters": {
                    "transaction_cost_bps": "0",
                    "slippage_bps": "5",
                    "benchmark_symbol": "SPX",
                }
            },
            {},
            CheckGrade.FAIL,
        ),
        (
            CheckName.SLIPPAGE,
            {
                "parameters": {
                    "transaction_cost_bps": "5",
                    "slippage_bps": "0",
                    "benchmark_symbol": "SPX",
                }
            },
            {},
            CheckGrade.FAIL,
        ),
        (
            CheckName.LIQUIDITY,
            {},
            {"average_dollar_volume": {"AAA": Decimal("1000")}, "order_notional": Decimal("100")},
            CheckGrade.FAIL,
        ),
        (CheckName.CAPACITY, {}, {"capacity_notional": None}, CheckGrade.FAIL),
        (CheckName.BENCHMARK_SELECTION, {}, {"benchmark_candidates": ["market"]}, CheckGrade.FAIL),
        (
            CheckName.ECONOMIC_PLAUSIBILITY,
            {"hypothesis_statement": "Guidance cuts are bad."},
            {},
            CheckGrade.FAIL,
        ),
        (
            CheckName.SELECTION_BIAS,
            {},
            {"eligible_symbols": ["AAA", "BBB"], "selection_rule": ""},
            CheckGrade.CONDITIONAL,
        ),
        (
            CheckName.MULTIPLE_TESTING,
            {},
            {"comparisons_tested": 3, "multiple_testing_adjustment": "bonferroni"},
            CheckGrade.CONDITIONAL,
        ),
        (
            CheckName.SAMPLE_SIZE,
            {"statistical_tests": {"test": {"observations": SAMPLE_PASS_AT - 1}}},
            {},
            CheckGrade.CONDITIONAL,
        ),
        (
            CheckName.TRANSACTION_COSTS,
            {
                "parameters": {
                    "transaction_cost_bps": "0.5",
                    "slippage_bps": "5",
                    "benchmark_symbol": "SPX",
                },
                "specification": {
                    "specification": {
                        "benchmark": "SPX",
                        "transaction_cost_bps": 0.5,
                        "slippage_bps": 5,
                        "event": {"type": "guidance_cut", "threshold": 8},
                    }
                },
            },
            {},
            CheckGrade.CONDITIONAL,
        ),
    ],
)
def test_each_methodological_check(
    check: CheckName,
    snapshot_changes: dict[str, object],
    audit_changes: dict[str, object],
    grade: CheckGrade,
) -> None:
    findings = _grades(_snapshot(**snapshot_changes), _audit(**audit_changes))
    assert findings[check].grade is grade
    assert findings[check].evidence
    assert findings[check].reason
    others = [item.grade for name, item in findings.items() if name is not check]
    if grade is CheckGrade.FAIL:
        assert verdict_for([item.grade for item in findings.values()]) is Verdict.FAIL
    else:
        assert all(item is CheckGrade.PASS for item in others)
        assert verdict_for([item.grade for item in findings.values()]) is Verdict.CONDITIONAL_PASS


def test_model_cannot_change_a_failing_grade(session: Session) -> None:
    _store_experiment(session)
    audit = _audit(capacity_notional=None)
    findings = grade_checks(EXPERIMENT_ID, _snapshot_from_row(session), audit)
    agent, model = _agent(session, _notes(findings))
    report = agent.review(EXPERIMENT_ID, audit)
    capacity = next(item for item in report.checks if item.check is CheckName.CAPACITY)
    assert capacity.grade is CheckGrade.FAIL
    assert report.verdict is Verdict.FAIL
    assert "PASS" not in model.requests[0].system_prompt.split("Return")[-1]


def test_evidence_ids_are_attributed(session: Session) -> None:
    _store_experiment(session)
    audit = _audit()
    findings = grade_checks(EXPERIMENT_ID, _snapshot_from_row(session), audit)
    agent, model = _agent(session, _notes(findings))
    report = agent.review(EXPERIMENT_ID, audit)
    prompt = model.requests[0].user_prompt
    for finding in findings:
        for evidence in finding.evidence:
            assert str(evidence.id) in prompt
        stored = next(item for item in report.checks if item.check is finding.check)
        assert stored.evidence_ids == [item.id for item in finding.evidence]
    foreign = _notes(findings).replace(
        str(findings[0].evidence[0].id),
        "00000000-0000-4000-8000-0000000000ff",
    )
    rejected, _ = _agent(session, foreign)
    with pytest.raises(UnsupportedEvidence):
        rejected.review(EXPERIMENT_ID, audit)
    assert _artifact_count(session) == 1


def test_structured_report_uses_provider_metadata(session: Session) -> None:
    _store_experiment(session)
    audit = _audit()
    findings = grade_checks(EXPERIMENT_ID, _snapshot_from_row(session), audit)
    agent, model = _agent(session, _notes(findings))
    report = agent.review(EXPERIMENT_ID, audit)
    assert report.verdict is Verdict.PASS
    assert [item.check for item in report.checks] == list(CheckName)
    assert report.model_name == "mock"
    assert report.model_version == "test-1"
    assert report.prompt_version == PROMPT_VERSION
    assert model.requests[0].prompt_version == PROMPT_VERSION
    assert "Do not defend the hypothesis." in model.requests[0].system_prompt
    artifact = session.scalars(select(ResearchArtifact)).one()
    assert artifact.artifact_type == "SKEPTIC_REPORT"
    assert artifact.experiment_id == EXPERIMENT_ID
    assert artifact.model_name == "mock"
    assert artifact.prompt_version == PROMPT_VERSION
    body = json.loads(artifact.content)
    assert body["output"]["verdict"] == "PASS"
    assert body["input"]["audit"]["selected_symbols"] == ["AAA"]
    assert body["evidence"]


def test_failure_handling_writes_nothing(session: Session) -> None:
    missing, missing_model = _agent(session, "{}")
    with pytest.raises(SkepticRejected):
        missing.review(EXPERIMENT_ID, _audit())
    assert missing_model.requests == []
    _store_experiment(session, status="LOCKED", completed_at=None)
    sealed, sealed_model = _agent(session, "{}")
    with pytest.raises(SkepticRejected):
        sealed.review(EXPERIMENT_ID, _audit())
    assert sealed_model.requests == []
    session.rollback()
    _store_experiment(session)
    audit = _audit()
    findings = grade_checks(EXPERIMENT_ID, _snapshot_from_row(session), audit)
    broken, _ = _agent(session, "not json")
    with pytest.raises(MalformedModelResponse):
        broken.review(EXPERIMENT_ID, audit)
    partial, _ = _agent(session, json.dumps({"notes": []}))
    with pytest.raises(MalformedModelResponse):
        partial.review(EXPERIMENT_ID, audit)
    defended, _ = _agent(session, _notes(findings, "The hypothesis is supported."))
    with pytest.raises(HypothesisDefense):
        defended.review(EXPERIMENT_ID, audit)
    failed, _ = _agent(session, _notes(findings), error=ModelProviderError("down"))
    with pytest.raises(ModelProviderError):
        failed.review(EXPERIMENT_ID, audit)
    assert _artifact_count(session) == 0
    row = session.get(Experiment, EXPERIMENT_ID)
    assert row is not None
    assert row.metrics == {"test": {"sharpe": "1"}}


def test_identical_mock_responses_replay_one_report(session: Session) -> None:
    _store_experiment(session)
    audit = _audit()
    findings = grade_checks(EXPERIMENT_ID, _snapshot_from_row(session), audit)
    agent, _ = _agent(session, _notes(findings))
    first = agent.review(EXPERIMENT_ID, audit)
    second = agent.review(EXPERIMENT_ID, audit)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert _artifact_count(session) == 1
    stored = session.get(Experiment, EXPERIMENT_ID)
    assert stored is not None
    assert stored.status == "TEST_RECORDED"
    assert stored.metrics == {"test": {"sharpe": "1"}}


def _notes(findings: list, challenge: str | None = None) -> str:
    text = challenge or (
        "A remaining objection is the chance this record fails to falsify the result."
    )
    return json.dumps(
        {
            "notes": [
                {
                    "check": item.check.value,
                    "evidence_ids": [str(evidence.id) for evidence in item.evidence],
                    "challenge": text,
                }
                for item in findings
            ]
        }
    )


def _agent(
    session: Session,
    content: str,
    *,
    error: Exception | None = None,
) -> tuple[SkepticAgent, MockLanguageModel]:
    model = MockLanguageModel(content, error=error)
    return SkepticAgent(session, model), model


def _artifact_count(session: Session) -> int:
    count = session.scalar(select(func.count()).select_from(ResearchArtifact))
    assert count is not None
    return count


def _snapshot_from_row(session: Session) -> ExperimentSnapshot:
    row = session.get(Experiment, EXPERIMENT_ID)
    hypothesis = session.get(Hypothesis, HYPOTHESIS_ID)
    assert row is not None and hypothesis is not None
    return ExperimentSnapshot(
        data_cutoff=row.data_cutoff,
        train_start=row.train_start,  # type: ignore[arg-type]
        train_end=row.train_end,  # type: ignore[arg-type]
        test_start=row.test_start,  # type: ignore[arg-type]
        test_end=row.test_end,  # type: ignore[arg-type]
        specification=row.specification if isinstance(row.specification, dict) else {},
        parameters=row.parameters if isinstance(row.parameters, dict) else {},
        statistical_tests=row.statistical_tests if isinstance(row.statistical_tests, dict) else {},
        hypothesis_statement=hypothesis.statement,
        hypothesis_mechanism=hypothesis.mechanism,
    )


def _store_experiment(
    session: Session,
    *,
    status: str = "TEST_RECORDED",
    completed_at: datetime | None = CUTOFF,
) -> None:
    session.add(
        Hypothesis(
            id=HYPOTHESIS_ID,
            title="guidance_cut 5d",
            statement=STATEMENT,
            mechanism=MECHANISM,
            status="DRAFT",
            universe={"asset_class": "US_EQUITY"},
            entry_rule={"timing": "NEXT_CLOSE"},
            exit_rule={"holding_period_days": 5},
            expected_horizon=5,
        )
    )
    session.flush()
    snapshot = _snapshot()
    session.add(
        Experiment(
            id=EXPERIMENT_ID,
            hypothesis_id=HYPOTHESIS_ID,
            dataset_version="prices-2020-01",
            code_version="skeptic-test",
            data_cutoff=snapshot.data_cutoff,
            experiment_timestamp=CUTOFF,
            started_at=CUTOFF,
            completed_at=completed_at,
            train_start=snapshot.train_start,
            train_end=snapshot.train_end,
            validation_start=date(2020, 1, 7),
            validation_end=date(2020, 1, 8),
            test_start=snapshot.test_start,
            test_end=snapshot.test_end,
            specification=snapshot.specification,
            parameters=snapshot.parameters,
            metrics={"test": {"sharpe": "1"}},
            statistical_tests=snapshot.statistical_tests,
            status=status,
        )
    )
    session.flush()


def _admin(sql: str) -> None:
    engine = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        connection.execute(text(sql))
    engine.dispose()


@pytest.fixture(scope="module")
def migrated_skeptic_database() -> Iterator[None]:
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
def session(migrated_skeptic_database: None) -> Iterator[Session]:
    del migrated_skeptic_database
    engine = create_engine(TEST_URL)
    with Session(engine) as db:
        try:
            yield db
        finally:
            db.rollback()
    engine.dispose()
