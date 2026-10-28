"""Deterministic methodological checks. A language model does not assign these grades."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, TypeGuard
from uuid import UUID, uuid5

from schemas.skeptic import (
    AuditFacts,
    CheckGrade,
    CheckName,
    EvidenceItem,
    ExperimentSnapshot,
    Verdict,
)

_NAMESPACE = UUID("e8b6c5ef-5ec0-4000-8000-0000000000e6")
_VAGUE_BENCHMARKS = frozenset({"market", "the market", "stocks", "bad", "good"})
_VAGUE_STATEMENT = "guidance cuts are bad."

SAMPLE_FAIL_BELOW = 30
SAMPLE_PASS_AT = 100
OVERFIT_FAIL_AT = 10
_COST_PASS_AT = Decimal(1)

Observation = tuple[str, str]


@dataclass(frozen=True)
class Assessment:
    check: CheckName
    grade: CheckGrade
    reason: str
    evidence: tuple[EvidenceItem, ...]


def grade_checks(
    experiment_id: UUID,
    snapshot: ExperimentSnapshot,
    audit: AuditFacts,
) -> list[Assessment]:
    """Grade every method. The order is the CheckName order."""
    return [
        _look_ahead(experiment_id, snapshot, audit),
        _survivorship(experiment_id, snapshot, audit),
        _selection(experiment_id, audit),
        _leakage(experiment_id, audit),
        _multiple_testing(experiment_id, audit),
        _overfitting(experiment_id, audit),
        _sample_size(experiment_id, snapshot),
        _regimes(experiment_id, snapshot, audit),
        _friction(
            experiment_id,
            snapshot,
            CheckName.TRANSACTION_COSTS,
            "transaction_cost_bps",
            "transaction cost",
        ),
        _friction(
            experiment_id,
            snapshot,
            CheckName.SLIPPAGE,
            "slippage_bps",
            "slippage",
        ),
        _liquidity(experiment_id, audit),
        _capacity(experiment_id, audit),
        _benchmark(experiment_id, snapshot, audit),
        _plausibility(experiment_id, snapshot),
    ]


def verdict_for(grades: list[CheckGrade] | tuple[CheckGrade, ...]) -> Verdict:
    """One failed check fails the result. A conditional check cannot be a clean pass."""
    if any(grade is CheckGrade.FAIL for grade in grades):
        return Verdict.FAIL
    if any(grade is CheckGrade.CONDITIONAL for grade in grades):
        return Verdict.CONDITIONAL_PASS
    return Verdict.PASS


def _look_ahead(
    experiment_id: UUID,
    snapshot: ExperimentSnapshot,
    audit: AuditFacts,
) -> Assessment:
    cutoff = snapshot.data_cutoff
    if not _aware(cutoff):
        return _assessment(
            experiment_id,
            CheckName.LOOK_AHEAD_BIAS,
            CheckGrade.FAIL,
            "feature or cutoff timing uses information from after the decision",
            [("data_cutoff", "missing" if cutoff is None else "naive")],
        )
    observed: list[Observation] = []
    if snapshot.test_end > cutoff.date():
        observed.append(("test_end", snapshot.test_end.isoformat()))
    if not _aware(audit.decision_timestamp) or audit.decision_timestamp > cutoff:
        observed.append(("decision_timestamp", _stamp(audit.decision_timestamp)))
    decision = audit.decision_timestamp if _aware(audit.decision_timestamp) else None
    for index, feature in enumerate(audit.feature_timestamps):
        if not _aware(feature) or feature > cutoff or (decision is not None and feature > decision):
            observed.append((f"feature_timestamps[{index}]", _stamp(feature)))
    if observed:
        return _assessment(
            experiment_id,
            CheckName.LOOK_AHEAD_BIAS,
            CheckGrade.FAIL,
            "feature or cutoff timing uses information from after the decision",
            observed,
        )
    return _assessment(
        experiment_id,
        CheckName.LOOK_AHEAD_BIAS,
        CheckGrade.PASS,
        "no look-ahead timestamp was recorded",
        [("data_cutoff", _stamp(cutoff))],
    )


def _survivorship(
    experiment_id: UUID,
    snapshot: ExperimentSnapshot,
    audit: AuditFacts,
) -> Assessment:
    observed: list[Observation] = []
    if not audit.retained_delisted:
        observed.append(("retained_delisted", "false"))
    members = {member.symbol: member for member in audit.members}
    for symbol in audit.selected_symbols:
        member = members.get(symbol)
        if member is None:
            observed.append((f"members[{symbol}]", "absent"))
            continue
        if member.listed_on > snapshot.train_start:
            observed.append((f"members[{symbol}].listed_on", member.listed_on.isoformat()))
        if member.delisted_on is not None and member.delisted_on <= snapshot.test_end:
            observed.append((f"members[{symbol}].delisted_on", member.delisted_on.isoformat()))
    if observed:
        return _assessment(
            experiment_id,
            CheckName.SURVIVORSHIP_BIAS,
            CheckGrade.FAIL,
            "the membership record drops a name or keeps a name that was not listed",
            observed,
        )
    return _assessment(
        experiment_id,
        CheckName.SURVIVORSHIP_BIAS,
        CheckGrade.PASS,
        "membership covers every selected symbol across the sample",
        [("retained_delisted", "true")],
    )


def _selection(experiment_id: UUID, audit: AuditFacts) -> Assessment:
    eligible = set(audit.eligible_symbols)
    missing = [symbol for symbol in audit.selected_symbols if symbol not in eligible]
    if not audit.selected_symbols or missing:
        observed = [("selected_symbols", ",".join(audit.selected_symbols) or "empty")]
        return _assessment(
            experiment_id,
            CheckName.SELECTION_BIAS,
            CheckGrade.FAIL,
            "a selected symbol is outside the eligible universe",
            observed,
        )
    if set(audit.selected_symbols) != eligible and not audit.selection_rule.strip():
        return _assessment(
            experiment_id,
            CheckName.SELECTION_BIAS,
            CheckGrade.CONDITIONAL,
            "a subset was selected without a recorded rule",
            [("selection_rule", "blank")],
        )
    return _assessment(
        experiment_id,
        CheckName.SELECTION_BIAS,
        CheckGrade.PASS,
        "selection is the eligible universe or a recorded rule",
        [("selected_symbols", ",".join(audit.selected_symbols))],
    )


def _leakage(experiment_id: UUID, audit: AuditFacts) -> Assessment:
    window = audit.normalization_window
    if window != "train":
        return _assessment(
            experiment_id,
            CheckName.DATA_LEAKAGE,
            CheckGrade.FAIL,
            "normalization uses data outside the training window",
            [("normalization_window", window)],
        )
    return _assessment(
        experiment_id,
        CheckName.DATA_LEAKAGE,
        CheckGrade.PASS,
        "normalization is fit on the training window",
        [("normalization_window", window)],
    )


def _multiple_testing(experiment_id: UUID, audit: AuditFacts) -> Assessment:
    observed = [
        ("comparisons_tested", str(audit.comparisons_tested)),
        ("multiple_testing_adjustment", audit.multiple_testing_adjustment),
    ]
    if audit.comparisons_tested == 1:
        grade = CheckGrade.PASS
        reason = "a single comparison was tested"
    elif audit.multiple_testing_adjustment == "none":
        grade = CheckGrade.FAIL
        reason = "more than one comparison was tested without an adjustment"
    else:
        grade = CheckGrade.CONDITIONAL
        reason = "multiple comparisons were adjusted and remain a competing explanation"
    return _assessment(experiment_id, CheckName.MULTIPLE_TESTING, grade, reason, observed)


def _overfitting(experiment_id: UUID, audit: AuditFacts) -> Assessment:
    searched = audit.specifications_searched
    if searched >= OVERFIT_FAIL_AT:
        grade = CheckGrade.FAIL
        reason = "ten or more specifications were searched"
    elif searched > 1:
        grade = CheckGrade.CONDITIONAL
        reason = "more than one specification was searched"
    else:
        grade = CheckGrade.PASS
        reason = "one specification was searched"
    return _assessment(
        experiment_id,
        CheckName.OVERFITTING,
        grade,
        reason,
        [("specifications_searched", str(searched))],
    )


def _sample_size(experiment_id: UUID, snapshot: ExperimentSnapshot) -> Assessment:
    observations = _observations(snapshot.statistical_tests)
    if observations is None or observations < SAMPLE_FAIL_BELOW:
        grade = CheckGrade.FAIL
        reason = "test observations are missing or below 30"
    elif observations < SAMPLE_PASS_AT:
        grade = CheckGrade.CONDITIONAL
        reason = "test observations are below 100"
    else:
        grade = CheckGrade.PASS
        reason = "test observations meet the sample threshold"
    return _assessment(
        experiment_id,
        CheckName.SAMPLE_SIZE,
        grade,
        reason,
        [("test.observations", "missing" if observations is None else str(observations))],
    )


def _regimes(
    experiment_id: UUID,
    snapshot: ExperimentSnapshot,
    audit: AuditFacts,
) -> Assessment:
    train_names = _regime_names(audit, snapshot.train_start, snapshot.train_end)
    test_names = _regime_names(audit, snapshot.test_start, snapshot.test_end)
    observed = [
        ("train_regimes", ",".join(sorted(train_names)) or "none"),
        ("test_regimes", ",".join(sorted(test_names)) or "none"),
    ]
    if not test_names or not test_names <= train_names:
        grade = CheckGrade.FAIL
        reason = "the test window falls outside the regimes recorded in training"
    elif len(train_names | test_names) < 2:
        grade = CheckGrade.CONDITIONAL
        reason = "only one regime covers the train and test windows"
    else:
        grade = CheckGrade.PASS
        reason = "every test regime also appears in training"
    return _assessment(experiment_id, CheckName.REGIME_DEPENDENCE, grade, reason, observed)


def _friction(
    experiment_id: UUID,
    snapshot: ExperimentSnapshot,
    check: CheckName,
    field: str,
    label: str,
) -> Assessment:
    spec_value = _as_decimal(_spec_body(snapshot.specification).get(field))
    param_value = _as_decimal(snapshot.parameters.get(field))
    observed = [
        (f"specification.{field}", _number(spec_value)),
        (f"parameters.{field}", _number(param_value)),
    ]
    if spec_value is None or param_value is None or spec_value != param_value or spec_value <= 0:
        grade = CheckGrade.FAIL
        reason = f"{label} is missing, zero, or disagrees across the record"
    elif spec_value < _COST_PASS_AT:
        grade = CheckGrade.CONDITIONAL
        reason = f"{label} is below one basis point"
    else:
        grade = CheckGrade.PASS
        reason = f"{label} is explicit and at least one basis point"
    return _assessment(experiment_id, check, grade, reason, observed)


def _liquidity(experiment_id: UUID, audit: AuditFacts) -> Assessment:
    observed: list[Observation] = [("order_notional", str(audit.order_notional))]
    stressed = False
    for symbol in audit.selected_symbols:
        volume = audit.average_dollar_volume.get(symbol)
        observed.append((f"average_dollar_volume[{symbol}]", _number(volume)))
        participation = audit.participation_limit
        if volume is None or volume <= 0 or audit.order_notional / volume > participation:
            stressed = True
    if not audit.selected_symbols or stressed:
        grade = CheckGrade.FAIL
        reason = "dollar volume is missing or the order exceeds the participation limit"
    elif audit.order_notional == 0:
        grade = CheckGrade.CONDITIONAL
        reason = "order notional is zero, so liquidity was not stressed"
    else:
        grade = CheckGrade.PASS
        reason = "the order is within the participation limit of recorded dollar volume"
    return _assessment(experiment_id, CheckName.LIQUIDITY, grade, reason, observed)


def _capacity(experiment_id: UUID, audit: AuditFacts) -> Assessment:
    capacity = audit.capacity_notional
    observed = [("capacity_notional", "missing" if capacity is None else str(capacity))]
    if capacity is None:
        grade = CheckGrade.FAIL
        reason = "capacity was not estimated"
    elif capacity < audit.order_notional:
        grade = CheckGrade.CONDITIONAL
        reason = "estimated capacity is below the order notional"
    else:
        grade = CheckGrade.PASS
        reason = "estimated capacity covers the order notional"
    return _assessment(experiment_id, CheckName.CAPACITY, grade, reason, observed)


def _benchmark(
    experiment_id: UUID,
    snapshot: ExperimentSnapshot,
    audit: AuditFacts,
) -> Assessment:
    spec_benchmark = str(_spec_body(snapshot.specification).get("benchmark", "")).strip()
    param_benchmark = str(snapshot.parameters.get("benchmark_symbol", "")).strip()
    candidates = [item.strip() for item in audit.benchmark_candidates]
    observed = [
        ("specification.benchmark", spec_benchmark or "blank"),
        ("parameters.benchmark_symbol", param_benchmark or "blank"),
        ("benchmark_candidates", ",".join(candidates) or "empty"),
    ]
    vague = _vague_benchmark(spec_benchmark) or _vague_benchmark(param_benchmark)
    vague = vague or any(_vague_benchmark(item) for item in candidates)
    distinct = {item for item in candidates if item}
    if (
        not spec_benchmark
        or not param_benchmark
        or spec_benchmark != param_benchmark
        or vague
        or not candidates
    ):
        grade = CheckGrade.FAIL
        reason = "benchmark is missing, vague, or disagrees across the record"
    elif spec_benchmark not in distinct or len(distinct) < 2:
        grade = CheckGrade.CONDITIONAL
        reason = "no alternative benchmark was recorded"
    else:
        grade = CheckGrade.PASS
        reason = "a specific benchmark was chosen from more than one candidate"
    return _assessment(experiment_id, CheckName.BENCHMARK_SELECTION, grade, reason, observed)


def _plausibility(experiment_id: UUID, snapshot: ExperimentSnapshot) -> Assessment:
    statement = snapshot.hypothesis_statement.strip()
    mechanism = (snapshot.hypothesis_mechanism or "").strip()
    event = str(_event_type(_spec_body(snapshot.specification))).strip()
    observed = [
        ("hypothesis_statement", statement or "blank"),
        ("hypothesis_mechanism", mechanism or "blank"),
        ("event.type", event or "blank"),
    ]
    specific = (
        statement.lower() != _VAGUE_STATEMENT
        and len(statement.split()) >= 8
        and any(character.isdigit() for character in statement)
    )
    named = bool(event) and event.lower() in mechanism.lower() and len(mechanism) >= 20
    if specific and named:
        grade = CheckGrade.PASS
        reason = "the mechanism names the event and the statement is specific"
    else:
        grade = CheckGrade.FAIL
        reason = "the statement or mechanism is too vague to identify a falsifiable channel"
    return _assessment(experiment_id, CheckName.ECONOMIC_PLAUSIBILITY, grade, reason, observed)


def _assessment(
    experiment_id: UUID,
    check: CheckName,
    grade: CheckGrade,
    reason: str,
    observed: list[Observation],
) -> Assessment:
    evidence = tuple(
        EvidenceItem(
            id=uuid5(_NAMESPACE, f"{experiment_id}|{check.value}|{index}|{field}|{value}"),
            check=check,
            field=field,
            observed=value,
        )
        for index, (field, value) in enumerate(observed)
    )
    return Assessment(check=check, grade=grade, reason=reason, evidence=evidence)


def _regime_names(audit: AuditFacts, start: date, end: date) -> set[str]:
    return {
        regime.name.strip()
        for regime in audit.regimes
        if regime.name.strip() and regime.start <= end and regime.end >= start
    }


def _observations(tests: dict[str, Any]) -> int | None:
    period = tests.get("test")
    if not isinstance(period, dict) or "observations" not in period:
        return None
    value = period["observations"]
    if isinstance(value, bool) or isinstance(value, float) or not isinstance(value, int | str):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _spec_body(specification: dict[str, Any]) -> dict[str, Any]:
    inner = specification.get("specification")
    if isinstance(inner, dict):
        return inner
    return specification


def _event_type(spec: dict[str, Any]) -> str:
    event = spec.get("event")
    if isinstance(event, dict):
        return str(event.get("type", ""))
    return ""


def _as_decimal(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            return None
        value = str(value)
    if not isinstance(value, int | str | Decimal):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    return number


def _aware(value: datetime | None) -> TypeGuard[datetime]:
    return value is not None and value.tzinfo is not None and value.utcoffset() is not None


def _stamp(value: datetime | None) -> str:
    if value is None:
        return "missing"
    return value.isoformat()


def _number(value: Decimal | None) -> str:
    if value is None:
        return "missing"
    return str(value)


def _vague_benchmark(value: str) -> bool:
    return not value or value.lower() in _VAGUE_BENCHMARKS
