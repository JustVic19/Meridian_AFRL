"""Validation engine. The policy chooses every test, and a stored result is never rewritten."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from uuid import UUID, uuid5

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.models import Experiment
from api.models import ValidationResult as ValidationResultRow
from api.models import ValidationRun as ValidationRunRow
from financial.performance import DISCLAIMER
from financial.precision import q
from research.backtest.engine import (
    BacktestError,
    BacktestRequest,
    BacktestResult,
    Bar,
    Signal,
    run_backtest,
)
from schemas.validation import VALIDATION_TEST_NAMES, ValidationStatus, ValidationTest

_NAMESPACE = UUID("a9d8e7f1-5ec0-4000-8000-0000000000f8")
CURRENT_POLICY_VERSION = "validation-1"


class ValidationRejected(Exception):
    """The run would skip a required test or rewrite a stored result."""


@dataclass(frozen=True)
class ValidationPolicy:
    """Versioned parameters. The test sequence is the registered order, not an argument."""

    version: str
    cost_stress_bps: Decimal
    threshold_shift: Decimal
    holding_shift: int

    @property
    def sequence(self) -> tuple[ValidationTest, ...]:
        return tuple(ValidationTest)


@dataclass(frozen=True)
class Observation:
    symbol: str
    session: date
    value: Decimal
    sector: str


@dataclass(frozen=True)
class RegimeLabel:
    name: str
    start: date
    end: date


@dataclass(frozen=True)
class ValidationBook:
    bars: tuple[Bar, ...]
    observations: tuple[Observation, ...]
    regimes: tuple[RegimeLabel, ...]


@dataclass(frozen=True)
class ValidationResult:
    id: UUID
    sequence_index: int
    test: ValidationTest
    status: ValidationStatus
    metrics: dict[str, object] | None
    detail: dict[str, object]


@dataclass(frozen=True)
class ValidationRun:
    id: UUID
    experiment_id: UUID
    policy: ValidationPolicy
    status: ValidationStatus
    executed_at: datetime
    results: tuple[ValidationResult, ...]


@dataclass(frozen=True)
class ComparisonRow:
    test: ValidationTest
    left_status: ValidationStatus
    right_status: ValidationStatus
    left_equity_end: str | None
    right_equity_end: str | None
    equity_end_delta: str | None


@dataclass(frozen=True)
class ValidationComparison:
    policy_version: str
    left_experiment_id: UUID
    right_experiment_id: UUID
    tests: tuple[ComparisonRow, ...]


_POLICIES = {
    CURRENT_POLICY_VERSION: ValidationPolicy(
        version=CURRENT_POLICY_VERSION,
        cost_stress_bps=Decimal(10),
        threshold_shift=Decimal(1),
        holding_shift=1,
    )
}


def policy_for(version: str) -> ValidationPolicy:
    """Return one registered policy. Unknown versions are not a way to drop a test."""
    found = _POLICIES.get(version)
    if found is None:
        raise ValidationRejected("validation policy is not registered")
    return found


class ValidationEngine:
    def __init__(self, session: Session) -> None:
        self._session = session

    def run(
        self,
        experiment_id: UUID,
        book: ValidationBook,
        *,
        executed_at: datetime,
        policy_version: str = CURRENT_POLICY_VERSION,
    ) -> ValidationRun:
        """Run the whole policy. A second call with the same inputs returns the stored run."""
        _require_aware(executed_at)
        policy = policy_for(policy_version)
        experiment = self._require_experiment(experiment_id)
        fresh = _evaluate(experiment, book, policy)
        run_id = uuid5(_NAMESPACE, f"{experiment_id}|{policy.version}")
        existing = self._session.get(ValidationRunRow, run_id)
        if existing is not None:
            stored = _load(self._session, existing)
            if not _same(stored, fresh):
                raise ValidationRejected("validation results are immutable")
            return stored
        status = _run_status(fresh)
        self._session.add(
            ValidationRunRow(
                id=run_id,
                experiment_id=experiment_id,
                policy_version=policy.version,
                policy=_policy_document(policy),
                status=status.value,
                executed_at=executed_at,
            )
        )
        self._session.flush()
        for result in fresh:
            self._session.add(
                ValidationResultRow(
                    id=result.id,
                    run_id=run_id,
                    sequence_index=result.sequence_index,
                    test_name=result.test.value,
                    status=result.status.value,
                    metrics=result.metrics,
                    detail=result.detail,
                )
            )
        self._session.flush()
        row = self._session.get(ValidationRunRow, run_id)
        if row is None:
            raise ValidationRejected("validation run was not stored")
        return _load(self._session, row)

    def compare(
        self,
        left_experiment_id: UUID,
        right_experiment_id: UUID,
        *,
        policy_version: str = CURRENT_POLICY_VERSION,
    ) -> ValidationComparison:
        """Align two stored runs. The policy version has to be the one that produced both."""
        policy = policy_for(policy_version)
        left = self._stored(left_experiment_id, policy.version)
        right = self._stored(right_experiment_id, policy.version)
        rows = tuple(
            _compare_row(left_result, right_result)
            for left_result, right_result in zip(left.results, right.results, strict=True)
        )
        return ValidationComparison(
            policy_version=policy.version,
            left_experiment_id=left_experiment_id,
            right_experiment_id=right_experiment_id,
            tests=rows,
        )

    def _require_experiment(self, experiment_id: UUID) -> Experiment:
        row = self._session.get(Experiment, experiment_id)
        if row is None:
            raise ValidationRejected("experiment does not exist")
        if row.status != "TEST_RECORDED" or row.completed_at is None:
            raise ValidationRejected("test results are not recorded")
        return row

    def _stored(self, experiment_id: UUID, policy_version: str) -> ValidationRun:
        row = self._session.scalar(
            select(ValidationRunRow).where(
                ValidationRunRow.experiment_id == experiment_id,
                ValidationRunRow.policy_version == policy_version,
            )
        )
        if row is None:
            raise ValidationRejected("validation run does not exist")
        return _load(self._session, row)


def _evaluate(
    experiment: Experiment,
    book: ValidationBook,
    policy: ValidationPolicy,
) -> tuple[ValidationResult, ...]:
    context = _context(experiment)
    results: list[ValidationResult] = []
    for index, test in enumerate(policy.sequence):
        results.append(_one(experiment.id, context, book, policy, test, index))
    return tuple(results)


def _one(
    experiment_id: UUID,
    context: _Context,
    book: ValidationBook,
    policy: ValidationPolicy,
    test: ValidationTest,
    index: int,
) -> ValidationResult:
    if test is ValidationTest.BASELINE:
        return _single(experiment_id, context, book, test, index, context.baseline, context.cost)
    if test is ValidationTest.TRANSACTION_COST:
        return _single(
            experiment_id,
            context,
            book,
            test,
            index,
            context.baseline,
            context.cost + policy.cost_stress_bps,
        )
    if test is ValidationTest.OUT_OF_SAMPLE:
        return _single(experiment_id, context, book, test, index, context.test, context.cost)
    if test is ValidationTest.ALTERNATIVE_THRESHOLD:
        return _single(
            experiment_id,
            context,
            book,
            test,
            index,
            context.test,
            context.cost,
            context.holding,
            context.threshold + policy.threshold_shift,
        )
    if test is ValidationTest.ALTERNATIVE_HOLDING_PERIOD:
        return _single(
            experiment_id,
            context,
            book,
            test,
            index,
            context.test,
            context.cost,
            context.holding + policy.holding_shift,
            context.threshold,
        )
    if test is ValidationTest.SUBPERIOD:
        return _subperiod(experiment_id, context, book, index)
    if test is ValidationTest.REGIME:
        return _regimes(experiment_id, context, book, index)
    return _sectors(experiment_id, context, book, index)


def _single(
    experiment_id: UUID,
    context: _Context,
    book: ValidationBook,
    test: ValidationTest,
    index: int,
    window: tuple[date, date],
    cost: Decimal,
    holding: int | None = None,
    threshold: Decimal | None = None,
) -> ValidationResult:
    used_holding = context.holding if holding is None else holding
    used_threshold = context.threshold if threshold is None else threshold
    metrics, error, count = _simulate(
        context, book, window, cost, used_holding, used_threshold, None, None
    )
    status = ValidationStatus.FAILED if error is not None else ValidationStatus.COMPLETE
    detail = _detail(context, cost, used_holding, used_threshold, window, error, count, [])
    return _result(experiment_id, index, test, status, metrics, detail)


def _subperiod(
    experiment_id: UUID,
    context: _Context,
    book: ValidationBook,
    index: int,
) -> ValidationResult:
    sessions = _sessions(book, context.test, context.benchmark, None)
    if len(sessions) < 2:
        detail = _detail(
            context,
            context.cost,
            context.holding,
            context.threshold,
            context.test,
            "test window is too short to split",
            0,
            [],
        )
        return _result(
            experiment_id, index, ValidationTest.SUBPERIOD, ValidationStatus.FAILED, None, detail
        )
    midpoint = len(sessions) // 2
    halves = (("first", sessions[:midpoint]), ("second", sessions[midpoint:]))
    parts = [
        _part(context, book, (dates[0], dates[-1]), name, None, set(dates))
        for name, dates in halves
    ]
    return _multi(experiment_id, context, ValidationTest.SUBPERIOD, index, parts)


def _regimes(
    experiment_id: UUID,
    context: _Context,
    book: ValidationBook,
    index: int,
) -> ValidationResult:
    overlaps = []
    for regime in book.regimes:
        start = max(regime.start, context.test[0])
        end = min(regime.end, context.test[1])
        if regime.name.strip() and start <= end:
            overlaps.append((regime.name.strip(), (start, end)))
    if not overlaps:
        detail = _detail(
            context,
            context.cost,
            context.holding,
            context.threshold,
            context.test,
            "no regime covers the test window",
            0,
            [],
        )
        return _result(
            experiment_id, index, ValidationTest.REGIME, ValidationStatus.FAILED, None, detail
        )
    parts = [_part(context, book, window, name, None, None) for name, window in sorted(overlaps)]
    return _multi(experiment_id, context, ValidationTest.REGIME, index, parts)


def _sectors(
    experiment_id: UUID,
    context: _Context,
    book: ValidationBook,
    index: int,
) -> ValidationResult:
    grouped: dict[str, set[str]] = {}
    for item in _eligible(book, context.test, context.threshold, context.benchmark, None, None):
        grouped.setdefault(item.sector.strip(), set()).add(item.symbol)
    if not grouped:
        detail = _detail(
            context,
            context.cost,
            context.holding,
            context.threshold,
            context.test,
            "no sector covers the test window",
            0,
            [],
        )
        return _result(
            experiment_id, index, ValidationTest.SECTOR, ValidationStatus.FAILED, None, detail
        )
    parts: list[dict[str, object]] = []
    for sector in sorted(grouped):
        if not sector:
            parts.append(
                {
                    "name": "",
                    "status": ValidationStatus.FAILED.value,
                    "metrics": None,
                    "error": "symbol is missing a sector",
                    "window": [_iso(context.test[0]), _iso(context.test[1])],
                    "signal_count": len(grouped[sector]),
                }
            )
            continue
        parts.append(_part(context, book, context.test, sector, grouped[sector], None))
    return _multi(experiment_id, context, ValidationTest.SECTOR, index, parts)


def _part(
    context: _Context,
    book: ValidationBook,
    window: tuple[date, date],
    name: str,
    symbols: set[str] | None,
    sessions: set[date] | None,
) -> dict[str, object]:
    metrics, error, count = _simulate(
        context,
        book,
        window,
        context.cost,
        context.holding,
        context.threshold,
        symbols,
        sessions,
    )
    return {
        "name": name,
        "status": ValidationStatus.FAILED.value if error else ValidationStatus.COMPLETE.value,
        "metrics": metrics,
        "error": error,
        "window": [_iso(window[0]), _iso(window[1])],
        "signal_count": count,
    }


def _multi(
    experiment_id: UUID,
    context: _Context,
    test: ValidationTest,
    index: int,
    parts: list[dict[str, object]],
) -> ValidationResult:
    failed = [part for part in parts if part["status"] != ValidationStatus.COMPLETE.value]
    error = str(failed[0]["error"]) if failed else None
    count = sum(int(str(part["signal_count"])) for part in parts)
    detail = _detail(
        context,
        context.cost,
        context.holding,
        context.threshold,
        context.test,
        error,
        count,
        parts,
    )
    status = ValidationStatus.FAILED if failed else ValidationStatus.COMPLETE
    return _result(experiment_id, index, test, status, None, detail)


def _simulate(
    context: _Context,
    book: ValidationBook,
    window: tuple[date, date],
    cost: Decimal,
    holding: int,
    threshold: Decimal,
    symbols: set[str] | None,
    sessions: set[date] | None,
) -> tuple[dict[str, object] | None, str | None, int]:
    eligible = _eligible(book, window, threshold, context.benchmark, symbols, sessions)
    bars = _bars(book, window, context.benchmark, symbols, sessions)
    signals = tuple(Signal(item.symbol, item.session) for item in eligible)
    try:
        result = run_backtest(
            BacktestRequest(
                initial_cash=context.cash,
                bars=bars,
                signals=signals,
                benchmark_symbol=context.benchmark,
                entry_timing=context.entry_timing,
                holding_period_sessions=holding,
                sizing_fraction=context.fraction,
                transaction_cost_bps=cost,
                slippage_bps=context.slippage,
                periods_per_year=context.periods_per_year,
                risk_free_rate_per_period=context.risk_free,
            )
        )
    except BacktestError as exc:
        return None, str(exc), len(signals)
    return _metrics(result), None, len(signals)


def _eligible(
    book: ValidationBook,
    window: tuple[date, date],
    threshold: Decimal,
    benchmark: str,
    symbols: set[str] | None,
    sessions: set[date] | None,
) -> tuple[Observation, ...]:
    chosen: list[Observation] = []
    seen: set[tuple[str, date]] = set()
    for item in book.observations:
        if item.session < window[0] or item.session > window[1]:
            continue
        if sessions is not None and item.session not in sessions:
            continue
        if symbols is not None and item.symbol not in symbols:
            continue
        if item.symbol == benchmark or item.value < threshold:
            continue
        key = (item.symbol, item.session)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(item)
    return tuple(sorted(chosen, key=lambda item: (item.session, item.symbol)))


def _bars(
    book: ValidationBook,
    window: tuple[date, date],
    benchmark: str,
    symbols: set[str] | None,
    sessions: set[date] | None,
) -> tuple[Bar, ...]:
    chosen: list[Bar] = []
    for bar in book.bars:
        if bar.session < window[0] or bar.session > window[1]:
            continue
        if sessions is not None and bar.session not in sessions:
            continue
        if bar.symbol != benchmark and symbols is not None and bar.symbol not in symbols:
            continue
        chosen.append(bar)
    return tuple(chosen)


def _sessions(
    book: ValidationBook,
    window: tuple[date, date],
    benchmark: str,
    symbols: set[str] | None,
) -> tuple[date, ...]:
    found = {
        bar.session
        for bar in _bars(book, window, benchmark, symbols, None)
        if bar.symbol != benchmark
    }
    return tuple(sorted(found))


def _metrics(result: BacktestResult) -> dict[str, object]:
    sharpe = None if result.sharpe is None else str(result.sharpe)
    drawdown = None if result.maximum_drawdown is None else str(result.maximum_drawdown)
    return {
        "number_of_observations": result.number_of_observations,
        "equity_end": str(result.equity[-1]),
        "sharpe": sharpe,
        "maximum_drawdown": drawdown,
        "disclaimer": DISCLAIMER,
    }


def _detail(
    context: _Context,
    cost: Decimal,
    holding: int,
    threshold: Decimal | None,
    window: tuple[date, date] | None = None,
    error: str | None = None,
    signal_count: int = 0,
    parts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    bounds = context.baseline if window is None else window
    return {
        "executed": True,
        "window": [_iso(bounds[0]), _iso(bounds[1])],
        "transaction_cost_bps": str(cost),
        "slippage_bps": str(context.slippage),
        "holding_period_sessions": holding,
        "threshold": "missing" if threshold is None else str(threshold),
        "signal_count": signal_count,
        "error": error,
        "parts": [] if parts is None else parts,
    }


def _result(
    experiment_id: UUID,
    index: int,
    test: ValidationTest,
    status: ValidationStatus,
    metrics: dict[str, object] | None,
    detail: dict[str, object],
) -> ValidationResult:
    run_id = uuid5(_NAMESPACE, f"{experiment_id}|{CURRENT_POLICY_VERSION}")
    return ValidationResult(
        id=uuid5(_NAMESPACE, f"{run_id}|{test.value}"),
        sequence_index=index,
        test=test,
        status=status,
        metrics=metrics,
        detail=detail,
    )


@dataclass(frozen=True)
class _Context:
    benchmark: str
    cash: Decimal
    entry_timing: str
    holding: int
    fraction: Decimal
    cost: Decimal
    slippage: Decimal
    periods_per_year: int
    risk_free: Decimal
    threshold: Decimal
    baseline: tuple[date, date]
    test: tuple[date, date]


def _context(experiment: Experiment) -> _Context:
    try:
        return _bound(experiment)
    except KeyError as exc:
        raise ValidationRejected("experiment record is unreadable") from exc


def _bound(experiment: Experiment) -> _Context:
    parameters = _mapping(experiment.parameters)
    return _Context(
        benchmark=str(parameters["benchmark_symbol"]).strip(),
        cash=_decimal(parameters["initial_cash"]),
        entry_timing=str(parameters["entry_timing"]),
        holding=_integer(parameters["holding_period_sessions"]),
        fraction=_decimal(parameters["sizing_fraction"]),
        cost=_decimal(parameters["transaction_cost_bps"]),
        slippage=_decimal(parameters["slippage_bps"]),
        periods_per_year=_integer(parameters["periods_per_year"]),
        risk_free=_decimal(parameters.get("risk_free_rate_per_period", "0")),
        threshold=_threshold(experiment.specification),
        baseline=(_as_date(experiment.train_start), _as_date(experiment.validation_end)),
        test=(_as_date(experiment.test_start), _as_date(experiment.test_end)),
    )


def _threshold(specification: object) -> Decimal:
    body = _mapping(specification)
    inner = body.get("specification")
    spec = _mapping(inner) if isinstance(inner, dict) else body
    event = spec.get("event")
    if not isinstance(event, dict) or "threshold" not in event:
        raise ValidationRejected("event threshold is missing")
    return _decimal(event["threshold"])


def _load(session: Session, row: ValidationRunRow) -> ValidationRun:
    policy = _policy_from_document(row.policy)
    results = session.scalars(
        select(ValidationResultRow)
        .where(ValidationResultRow.run_id == row.id)
        .order_by(ValidationResultRow.sequence_index)
    ).all()
    loaded = tuple(_stored_result(item) for item in results)
    if [item.test.value for item in loaded] != list(VALIDATION_TEST_NAMES):
        raise ValidationRejected("stored validation sequence is incomplete")
    return ValidationRun(
        id=row.id,
        experiment_id=row.experiment_id,
        policy=policy,
        status=ValidationStatus(row.status),
        executed_at=row.executed_at,
        results=loaded,
    )


def _stored_result(row: ValidationResultRow) -> ValidationResult:
    metrics = row.metrics if isinstance(row.metrics, dict) else None
    detail = row.detail if isinstance(row.detail, dict) else {}
    return ValidationResult(
        id=row.id,
        sequence_index=row.sequence_index,
        test=ValidationTest(row.test_name),
        status=ValidationStatus(row.status),
        metrics=dict(metrics) if metrics is not None else None,
        detail=dict(detail),
    )


def _same(stored: ValidationRun, fresh: tuple[ValidationResult, ...]) -> bool:
    if len(stored.results) != len(fresh):
        return False
    for left, right in zip(stored.results, fresh, strict=True):
        if left.test is not right.test or left.status is not right.status:
            return False
        if _canonical(left.metrics) != _canonical(right.metrics):
            return False
        if _canonical(left.detail) != _canonical(right.detail):
            return False
    return True


def _compare_row(left: ValidationResult, right: ValidationResult) -> ComparisonRow:
    left_equity = _equity(left.metrics)
    right_equity = _equity(right.metrics)
    delta = None
    if left_equity is not None and right_equity is not None:
        delta = str(q(Decimal(right_equity) - Decimal(left_equity)))
    return ComparisonRow(
        test=left.test,
        left_status=left.status,
        right_status=right.status,
        left_equity_end=left_equity,
        right_equity_end=right_equity,
        equity_end_delta=delta,
    )


def _equity(metrics: dict[str, object] | None) -> str | None:
    if metrics is None:
        return None
    value = metrics.get("equity_end")
    if value is None:
        return None
    return str(value)


def _policy_document(policy: ValidationPolicy) -> dict[str, object]:
    return {
        "version": policy.version,
        "sequence": [test.value for test in policy.sequence],
        "cost_stress_bps": str(policy.cost_stress_bps),
        "threshold_shift": str(policy.threshold_shift),
        "holding_shift": policy.holding_shift,
    }


def _policy_from_document(document: object) -> ValidationPolicy:
    body = _mapping(document)
    version = str(body.get("version", ""))
    registered = policy_for(version)
    sequence = body.get("sequence")
    if sequence != [test.value for test in registered.sequence]:
        raise ValidationRejected("stored policy sequence does not match the registered policy")
    stored = ValidationPolicy(
        version=version,
        cost_stress_bps=_decimal(body.get("cost_stress_bps")),
        threshold_shift=_decimal(body.get("threshold_shift")),
        holding_shift=_integer(body.get("holding_shift")),
    )
    if stored != registered:
        raise ValidationRejected("stored policy does not match the registered policy")
    return stored


def _run_status(results: tuple[ValidationResult, ...]) -> ValidationStatus:
    if any(result.status is ValidationStatus.FAILED for result in results):
        return ValidationStatus.FAILED
    return ValidationStatus.COMPLETE


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    raise ValidationRejected("experiment record is unreadable")


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise ValidationRejected("experiment record is unreadable")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValidationRejected("experiment record is unreadable") from exc
    if not number.is_finite():
        raise ValidationRejected("experiment record is unreadable")
    return number


def _integer(value: object) -> int:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValidationRejected("experiment record is unreadable")
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValidationRejected("experiment record is unreadable") from exc


def _as_date(value: date | None) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise ValidationRejected("experiment period is missing")
    return value


def _iso(value: date) -> str:
    return value.isoformat()


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationRejected("timestamps must be timezone-aware")
