"""Experiment engine. A model never receives test results before the experiment is locked."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from uuid import UUID, uuid5

from sqlalchemy.orm import Session

from api.models import Experiment, Hypothesis
from research.backtest.engine import BacktestRequest, Bar, Signal, run_backtest
from research.validation.specification import ValidatedSpecification, validate
from schemas.experiment_spec import DatasetCatalog, ExperimentSpecification

_NAMESPACE = UUID("e7f4a3cd-5ec0-4000-8000-0000000000d4")


class ExperimentState(StrEnum):
    CREATED = "CREATED"
    TRAIN_RECORDED = "TRAIN_RECORDED"
    VALIDATION_RECORDED = "VALIDATION_RECORDED"
    LOCKED = "LOCKED"
    TEST_RECORDED = "TEST_RECORDED"


class Period(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


_NEXT_STATE = {
    Period.TRAIN: ExperimentState.TRAIN_RECORDED,
    Period.VALIDATION: ExperimentState.VALIDATION_RECORDED,
    Period.TEST: ExperimentState.TEST_RECORDED,
}
_REQUIRED_STATE = {
    Period.TRAIN: ExperimentState.CREATED,
    Period.VALIDATION: ExperimentState.TRAIN_RECORDED,
    Period.TEST: ExperimentState.LOCKED,
}


class ExperimentIntegrityError(Exception):
    """The requested action would leak or rewrite a sealed result."""


@dataclass(frozen=True)
class ExperimentParameters:
    initial_cash: Decimal
    entry_timing: str
    holding_period_sessions: int
    sizing_fraction: Decimal
    transaction_cost_bps: Decimal
    slippage_bps: Decimal
    periods_per_year: int
    benchmark_symbol: str
    signals: tuple[Signal, ...]
    risk_free_rate_per_period: Decimal = Decimal(0)


@dataclass(frozen=True)
class ResearchView:
    """What a researcher may see. Test metrics are absent until the test is recorded."""

    experiment_id: UUID
    hypothesis_id: UUID
    parent_experiment_id: UUID | None
    dataset_version: str
    code_version: str
    status: ExperimentState
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date
    specification: dict[str, object]
    parameters: dict[str, object]
    metrics: dict[str, object]
    statistical_tests: dict[str, object]
    started_at: datetime | None
    experiment_timestamp: datetime | None
    completed_at: datetime | None


class ExperimentEngine:
    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self,
        *,
        hypothesis_id: UUID,
        specification: ExperimentSpecification,
        catalog: DatasetCatalog,
        parameters: ExperimentParameters,
        executed_at: datetime,
        parent_experiment_id: UUID | None = None,
    ) -> ResearchView:
        """Open one experiment. The same inputs return the existing row."""
        _require_aware(executed_at)
        if self._session.get(Hypothesis, hypothesis_id) is None:
            raise ExperimentIntegrityError("hypothesis does not exist")
        validated = validate(specification, catalog)
        parent = None
        if parent_experiment_id is not None:
            parent = self._session.get(Experiment, parent_experiment_id)
        if parent_experiment_id is not None and parent is None:
            raise ExperimentIntegrityError("parent experiment does not exist")
        experiment_id = _experiment_id(hypothesis_id, validated, parameters, parent_experiment_id)
        existing = self._session.get(Experiment, experiment_id)
        if existing is not None:
            return self.research_view(experiment_id)
        self._session.add(
            Experiment(
                id=experiment_id,
                hypothesis_id=hypothesis_id,
                parent_experiment_id=parent_experiment_id,
                dataset_version=specification.dataset_version,
                code_version=specification.code_version,
                data_cutoff=specification.data_cutoff,
                experiment_timestamp=executed_at,
                started_at=executed_at,
                train_start=validated.train.start,
                train_end=validated.train.end,
                validation_start=validated.validation.start,
                validation_end=validated.validation.end,
                test_start=validated.test.start,
                test_end=validated.test.end,
                specification=_specification_document(validated),
                parameters=_parameters_document(parameters),
                metrics={},
                statistical_tests={},
                status=ExperimentState.CREATED.value,
            )
        )
        self._session.flush()
        return self.research_view(experiment_id)

    def record_train(self, experiment_id: UUID, bars: tuple[Bar, ...]) -> ResearchView:
        return self._record(experiment_id, Period.TRAIN, bars, None)

    def record_validation(self, experiment_id: UUID, bars: tuple[Bar, ...]) -> ResearchView:
        return self._record(experiment_id, Period.VALIDATION, bars, None)

    def lock(self, experiment_id: UUID) -> ResearchView:
        """Freeze the design. Test results still do not exist."""
        row = self._require(experiment_id)
        state = ExperimentState(row.status)
        if state is ExperimentState.LOCKED or state is ExperimentState.TEST_RECORDED:
            return self.research_view(experiment_id)
        if state is not ExperimentState.VALIDATION_RECORDED:
            raise ExperimentIntegrityError("experiment state does not allow this action")
        row.status = ExperimentState.LOCKED.value
        self._session.flush()
        return self.research_view(experiment_id)

    def record_test(
        self,
        experiment_id: UUID,
        bars: tuple[Bar, ...],
        *,
        executed_at: datetime,
    ) -> ResearchView:
        _require_aware(executed_at)
        return self._record(experiment_id, Period.TEST, bars, executed_at)

    def bars_for(
        self,
        experiment_id: UUID,
        period: Period,
        bars: tuple[Bar, ...],
    ) -> tuple[Bar, ...]:
        """Return one period. The test period is sealed until the experiment is locked."""
        row = self._require(experiment_id)
        if period is Period.TEST and ExperimentState(row.status) not in {
            ExperimentState.LOCKED,
            ExperimentState.TEST_RECORDED,
        }:
            raise ExperimentIntegrityError("test period is sealed")
        return _bars_in_window(bars, _bounds(row, period))

    def research_view(self, experiment_id: UUID) -> ResearchView:
        row = self._require(experiment_id)
        revealed = ExperimentState(row.status) is ExperimentState.TEST_RECORDED
        return ResearchView(
            experiment_id=row.id,
            hypothesis_id=row.hypothesis_id,
            parent_experiment_id=row.parent_experiment_id,
            dataset_version=row.dataset_version,
            code_version=row.code_version,
            status=ExperimentState(row.status),
            train_start=_date(row.train_start),
            train_end=_date(row.train_end),
            validation_start=_date(row.validation_start),
            validation_end=_date(row.validation_end),
            test_start=_date(row.test_start),
            test_end=_date(row.test_end),
            specification=_document(row.specification),
            parameters=_document(row.parameters),
            metrics=_visible(_document(row.metrics), revealed),
            statistical_tests=_visible(_document(row.statistical_tests), revealed),
            started_at=row.started_at,
            experiment_timestamp=row.experiment_timestamp,
            completed_at=row.completed_at,
        )

    def revise(
        self,
        experiment_id: UUID,
        *,
        hypothesis_id: UUID,
        specification: ExperimentSpecification,
        catalog: DatasetCatalog,
        parameters: ExperimentParameters,
        executed_at: datetime,
    ) -> ResearchView:
        """A changed design after test results is a child experiment. The parent stays put."""
        parent = self._require(experiment_id)
        if ExperimentState(parent.status) is not ExperimentState.TEST_RECORDED:
            raise ExperimentIntegrityError("test results have not been recorded")
        parent_metrics = json.dumps(_document(parent.metrics), sort_keys=True)
        child = self.create(
            hypothesis_id=hypothesis_id,
            specification=specification,
            catalog=catalog,
            parameters=parameters,
            executed_at=executed_at,
            parent_experiment_id=experiment_id,
        )
        unchanged = self._require(experiment_id)
        if json.dumps(_document(unchanged.metrics), sort_keys=True) != parent_metrics:
            raise ExperimentIntegrityError("recorded results are immutable")
        if unchanged.status != ExperimentState.TEST_RECORDED.value:
            raise ExperimentIntegrityError("recorded results are immutable")
        return child

    def lineage(self, experiment_id: UUID) -> tuple[UUID, ...]:
        """Root experiment first, requested experiment last."""
        chain: list[UUID] = []
        seen: set[UUID] = set()
        current: UUID | None = experiment_id
        while current is not None:
            if current in seen:
                raise ExperimentIntegrityError("experiment lineage is cyclic")
            seen.add(current)
            row = self._require(current)
            chain.append(row.id)
            current = row.parent_experiment_id
        chain.reverse()
        return tuple(chain)

    def _record(
        self,
        experiment_id: UUID,
        period: Period,
        bars: tuple[Bar, ...],
        executed_at: datetime | None,
    ) -> ResearchView:
        row = self._require(experiment_id)
        state = ExperimentState(row.status)
        if state is ExperimentState.TEST_RECORDED:
            self._require_replay(row, period, bars)
            return self.research_view(experiment_id)
        scoped = _bars_in_window(bars, _bounds(row, period))
        metrics, tests = _evaluate(_parameters(row.parameters), scoped)
        stored_metrics = _document(row.metrics)
        stored_tests = _document(row.statistical_tests)
        if period.value in stored_metrics:
            self._require_same(stored_metrics, stored_tests, period, metrics, tests)
            return self.research_view(experiment_id)
        if state is not _REQUIRED_STATE[period]:
            raise ExperimentIntegrityError("experiment state does not allow this action")
        stored_metrics[period.value] = metrics
        stored_tests[period.value] = tests
        row.metrics = stored_metrics
        row.statistical_tests = stored_tests
        row.status = _NEXT_STATE[period].value
        if period is Period.TEST:
            if executed_at is None:
                raise ExperimentIntegrityError("test results require a timestamp")
            row.completed_at = executed_at
        self._session.flush()
        return self.research_view(experiment_id)

    def _require_replay(
        self,
        row: Experiment,
        period: Period,
        bars: tuple[Bar, ...],
    ) -> None:
        scoped = _bars_in_window(bars, _bounds(row, period))
        metrics, tests = _evaluate(_parameters(row.parameters), scoped)
        self._require_same(
            _document(row.metrics),
            _document(row.statistical_tests),
            period,
            metrics,
            tests,
        )

    def _require_same(
        self,
        stored_metrics: dict[str, object],
        stored_tests: dict[str, object],
        period: Period,
        metrics: dict[str, object],
        tests: dict[str, object],
    ) -> None:
        if stored_metrics.get(period.value) != metrics or stored_tests.get(period.value) != tests:
            raise ExperimentIntegrityError("recorded results are immutable")

    def _require(self, experiment_id: UUID) -> Experiment:
        row = self._session.get(Experiment, experiment_id)
        if row is None:
            raise ExperimentIntegrityError("experiment does not exist")
        return row


def _experiment_id(
    hypothesis_id: UUID,
    validated: ValidatedSpecification,
    parameters: ExperimentParameters,
    parent_experiment_id: UUID | None,
) -> UUID:
    payload = {
        "hypothesis_id": str(hypothesis_id),
        "parent_experiment_id": None if parent_experiment_id is None else str(parent_experiment_id),
        "specification": _specification_document(validated),
        "parameters": _parameters_document(parameters),
    }
    return uuid5(_NAMESPACE, json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _specification_document(validated: ValidatedSpecification) -> dict[str, object]:
    return {
        "specification": validated.specification.model_dump(mode="json"),
        "train": [validated.train.start.isoformat(), validated.train.end.isoformat()],
        "validation": [
            validated.validation.start.isoformat(),
            validated.validation.end.isoformat(),
        ],
        "test": [validated.test.start.isoformat(), validated.test.end.isoformat()],
    }


def _parameters_document(parameters: ExperimentParameters) -> dict[str, object]:
    return {
        "initial_cash": str(parameters.initial_cash),
        "entry_timing": parameters.entry_timing,
        "holding_period_sessions": parameters.holding_period_sessions,
        "sizing_fraction": str(parameters.sizing_fraction),
        "transaction_cost_bps": str(parameters.transaction_cost_bps),
        "slippage_bps": str(parameters.slippage_bps),
        "periods_per_year": parameters.periods_per_year,
        "benchmark_symbol": parameters.benchmark_symbol,
        "risk_free_rate_per_period": str(parameters.risk_free_rate_per_period),
        "signals": [
            {"symbol": signal.symbol, "session": signal.session.isoformat()}
            for signal in parameters.signals
        ],
    }


def _parameters(document: object) -> ExperimentParameters:
    body = _document(document)
    signals = body["signals"]
    if not isinstance(signals, list):
        raise ExperimentIntegrityError("parameters are unreadable")
    return ExperimentParameters(
        initial_cash=Decimal(str(body["initial_cash"])),
        entry_timing=str(body["entry_timing"]),
        holding_period_sessions=int(str(body["holding_period_sessions"])),
        sizing_fraction=Decimal(str(body["sizing_fraction"])),
        transaction_cost_bps=Decimal(str(body["transaction_cost_bps"])),
        slippage_bps=Decimal(str(body["slippage_bps"])),
        periods_per_year=int(str(body["periods_per_year"])),
        benchmark_symbol=str(body["benchmark_symbol"]),
        risk_free_rate_per_period=Decimal(str(body["risk_free_rate_per_period"])),
        signals=tuple(
            Signal(str(item["symbol"]), date.fromisoformat(str(item["session"])))
            for item in signals
            if isinstance(item, dict)
        ),
    )


def _evaluate(
    parameters: ExperimentParameters,
    bars: tuple[Bar, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    start, end = _span(bars, parameters.benchmark_symbol)
    result = run_backtest(
        BacktestRequest(
            initial_cash=parameters.initial_cash,
            bars=bars,
            signals=tuple(
                signal for signal in parameters.signals if start <= signal.session <= end
            ),
            benchmark_symbol=parameters.benchmark_symbol,
            entry_timing=parameters.entry_timing,
            holding_period_sessions=parameters.holding_period_sessions,
            sizing_fraction=parameters.sizing_fraction,
            transaction_cost_bps=parameters.transaction_cost_bps,
            slippage_bps=parameters.slippage_bps,
            periods_per_year=parameters.periods_per_year,
            risk_free_rate_per_period=parameters.risk_free_rate_per_period,
        )
    )
    returns = [
        current / previous - 1
        for previous, current in zip(result.equity, result.equity[1:], strict=False)
    ]
    mean = Decimal(0) if not returns else sum(returns, Decimal(0)) / Decimal(len(returns))
    metrics = {
        "number_of_observations": result.number_of_observations,
        "equity_end": str(result.equity[-1]),
        "sharpe": None if result.sharpe is None else str(result.sharpe),
        "maximum_drawdown": None
        if result.maximum_drawdown is None
        else str(result.maximum_drawdown),
    }
    tests = {
        "observations": result.number_of_observations,
        "mean_return": str(mean.quantize(Decimal("0.00000001"))),
    }
    return metrics, tests


def _bars_in_window(bars: tuple[Bar, ...], bounds: tuple[date, date]) -> tuple[Bar, ...]:
    start, end = bounds
    return tuple(bar for bar in bars if start <= bar.session <= end)


def _bounds(row: Experiment, period: Period) -> tuple[date, date]:
    if period is Period.TRAIN:
        return _date(row.train_start), _date(row.train_end)
    if period is Period.VALIDATION:
        return _date(row.validation_start), _date(row.validation_end)
    return _date(row.test_start), _date(row.test_end)


def _span(bars: tuple[Bar, ...], benchmark: str) -> tuple[date, date]:
    sessions = [bar.session for bar in bars if bar.symbol != benchmark]
    if not sessions:
        raise ExperimentIntegrityError("the period has no prices")
    return min(sessions), max(sessions)


def _visible(document: dict[str, object], revealed: bool) -> dict[str, object]:
    if revealed:
        return document
    return {key: value for key, value in document.items() if key != Period.TEST.value}


def _document(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _date(value: date | None) -> date:
    if value is None:
        raise ExperimentIntegrityError("experiment period is missing")
    return value


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ExperimentIntegrityError("timestamps must be timezone-aware")
