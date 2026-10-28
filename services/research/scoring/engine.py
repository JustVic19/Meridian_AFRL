"""Research score. The number ranks a recorded experiment and does not authorize a trade."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid5

from sqlalchemy.orm import Session

from api.models import Experiment
from api.models import ResearchScore as ResearchScoreRow
from financial.precision import q
from schemas.scoring import SCORE_COMPONENTS, ComponentStatus, ScoreComponent, ScoreStatus
from schemas.skeptic import CheckGrade

_NAMESPACE = UUID("c2d3e4f5-5ec0-4000-8000-0000000000fa")
CURRENT_POLICY_VERSION = "research-score-1"
_PURPOSE = "research_ranking"
_DISCLAIMER = (
    "Research ranking only. This score does not authorize a live trade, an order, "
    "or a portfolio allocation."
)


class ScoreRejected(Exception):
    """The score would invent a missing metric, change a weight, or rewrite a stored score."""


@dataclass(frozen=True)
class ScoringPolicy:
    """Fixed weights and scales. Callers cannot replace either."""

    version: str
    weights: tuple[tuple[ScoreComponent, Decimal], ...]
    oos_sharpe_cap: Decimal
    confidence_floor: int
    confidence_ceiling: int
    conditional_credit: Decimal
    robustness_tests: int
    simplicity_ceiling: int
    regime_spread_cap: Decimal

    @property
    def purpose(self) -> str:
        return _PURPOSE

    @property
    def disclaimer(self) -> str:
        return _DISCLAIMER


@dataclass(frozen=True)
class Scorecard:
    """Measured inputs. A missing value stays missing."""

    oos_sharpe: Decimal | None = None
    robustness_completed: int | None = None
    test_observations: int | None = None
    mechanism: CheckGrade | None = None
    baseline_sharpe: Decimal | None = None
    stressed_cost_sharpe: Decimal | None = None
    transaction_cost_failed: bool = False
    regime_sharpes: tuple[Decimal, ...] | None = None
    regime_failed: bool = False
    capacity: CheckGrade | None = None
    specifications_searched: int | None = None


@dataclass(frozen=True)
class ComponentScore:
    component: ScoreComponent
    weight: Decimal
    status: ComponentStatus
    score: Decimal | None


@dataclass(frozen=True)
class ScoreEvaluation:
    status: ScoreStatus
    score: Decimal | None
    components: tuple[ComponentScore, ...]


@dataclass(frozen=True)
class ResearchScore:
    id: UUID
    experiment_id: UUID
    policy_version: str
    purpose: str
    status: ScoreStatus
    score: Decimal | None
    components: tuple[ComponentScore, ...]
    weights: tuple[tuple[ScoreComponent, Decimal], ...]
    disclaimer: str
    live_trade_authorized: bool
    executed_at: datetime


_WEIGHTS = (
    (ScoreComponent.OOS_PERFORMANCE, Decimal("0.20")),
    (ScoreComponent.ROBUSTNESS, Decimal("0.15")),
    (ScoreComponent.STATISTICAL_CONFIDENCE, Decimal("0.15")),
    (ScoreComponent.ECONOMIC_MECHANISM, Decimal("0.15")),
    (ScoreComponent.TRANSACTION_COST_RESILIENCE, Decimal("0.10")),
    (ScoreComponent.REGIME_STABILITY, Decimal("0.10")),
    (ScoreComponent.CAPACITY, Decimal("0.10")),
    (ScoreComponent.SIMPLICITY, Decimal("0.05")),
)

_POLICIES = {
    CURRENT_POLICY_VERSION: ScoringPolicy(
        version=CURRENT_POLICY_VERSION,
        weights=_WEIGHTS,
        oos_sharpe_cap=Decimal(1),
        confidence_floor=30,
        confidence_ceiling=100,
        conditional_credit=Decimal("0.5"),
        robustness_tests=4,
        simplicity_ceiling=10,
        regime_spread_cap=Decimal(1),
    )
}


def policy_for(version: str) -> ScoringPolicy:
    """Return one registered policy. A caller cannot supply a new set of weights."""
    found = _POLICIES.get(version)
    if found is None:
        raise ScoreRejected("scoring policy is not registered")
    if sum((weight for _, weight in found.weights), Decimal(0)) != Decimal("1.00"):
        raise ScoreRejected("scoring weights do not sum to one")
    if tuple(component.value for component, _weight in found.weights) != SCORE_COMPONENTS:
        raise ScoreRejected("scoring weights do not cover every component")
    return found


def evaluate(card: Scorecard, policy: ScoringPolicy) -> ScoreEvaluation:
    """Score one card. Missing inputs withhold the total instead of receiving a zero."""
    _require_card(card, policy)
    weights = dict(policy.weights)
    components = (
        _metric(ScoreComponent.OOS_PERFORMANCE, weights, _oos(card, policy)),
        _metric(ScoreComponent.ROBUSTNESS, weights, _robustness(card, policy)),
        _metric(ScoreComponent.STATISTICAL_CONFIDENCE, weights, _confidence(card, policy)),
        _metric(ScoreComponent.ECONOMIC_MECHANISM, weights, _grade(card.mechanism, policy)),
        _metric(ScoreComponent.TRANSACTION_COST_RESILIENCE, weights, _costs(card)),
        _metric(ScoreComponent.REGIME_STABILITY, weights, _regimes(card, policy)),
        _metric(ScoreComponent.CAPACITY, weights, _grade(card.capacity, policy)),
        _metric(ScoreComponent.SIMPLICITY, weights, _simplicity(card, policy)),
    )
    if any(item.status is ComponentStatus.MISSING for item in components):
        return ScoreEvaluation(ScoreStatus.INCOMPLETE, None, components)
    total = sum(
        (item.weight * item.score for item in components if item.score is not None),
        Decimal(0),
    )
    return ScoreEvaluation(ScoreStatus.SCORED, q(total), components)


class ResearchScoreEngine:
    def __init__(self, session: Session) -> None:
        self._session = session

    def score(
        self,
        experiment_id: UUID,
        card: Scorecard,
        *,
        executed_at: datetime,
        policy_version: str = CURRENT_POLICY_VERSION,
    ) -> ResearchScore:
        """Store one research ranking. The same card returns the stored score."""
        _require_aware(executed_at)
        policy = policy_for(policy_version)
        self._require_experiment(experiment_id)
        evaluation = evaluate(card, policy)
        score_id = uuid5(_NAMESPACE, f"{experiment_id}|{policy.version}")
        existing = self._session.get(ResearchScoreRow, score_id)
        if existing is not None:
            stored = _load(existing)
            if not _same(stored, evaluation, existing.inputs, card):
                raise ScoreRejected("research scores are immutable")
            return stored
        self._session.add(
            ResearchScoreRow(
                id=score_id,
                experiment_id=experiment_id,
                policy_version=policy.version,
                policy=_policy_document(policy),
                status=evaluation.status.value,
                score=None if evaluation.score is None else str(evaluation.score),
                components=_components_document(evaluation.components),
                weights=_weights_document(policy),
                inputs=_inputs_document(card),
                live_trade_authorized=False,
                executed_at=executed_at,
            )
        )
        self._session.flush()
        row = self._session.get(ResearchScoreRow, score_id)
        if row is None:
            raise ScoreRejected("research score was not stored")
        return _load(row)

    def _require_experiment(self, experiment_id: UUID) -> None:
        row = self._session.get(Experiment, experiment_id)
        if row is None:
            raise ScoreRejected("experiment does not exist")
        if row.status != "TEST_RECORDED" or row.completed_at is None:
            raise ScoreRejected("test results are not recorded")


def _oos(card: Scorecard, policy: ScoringPolicy) -> Decimal | None:
    if card.oos_sharpe is None:
        return None
    return _clamp(card.oos_sharpe / policy.oos_sharpe_cap)


def _robustness(card: Scorecard, policy: ScoringPolicy) -> Decimal | None:
    if card.robustness_completed is None:
        return None
    return Decimal(card.robustness_completed) / Decimal(policy.robustness_tests)


def _confidence(card: Scorecard, policy: ScoringPolicy) -> Decimal | None:
    observations = card.test_observations
    if observations is None:
        return None
    if observations <= policy.confidence_floor:
        return Decimal(0)
    if observations >= policy.confidence_ceiling:
        return Decimal(1)
    span = policy.confidence_ceiling - policy.confidence_floor
    return Decimal(observations - policy.confidence_floor) / Decimal(span)


def _grade(grade: CheckGrade | None, policy: ScoringPolicy) -> Decimal | None:
    if grade is None:
        return None
    if grade is CheckGrade.PASS:
        return Decimal(1)
    if grade is CheckGrade.CONDITIONAL:
        return policy.conditional_credit
    return Decimal(0)


def _costs(card: Scorecard) -> Decimal | None:
    if card.transaction_cost_failed:
        return Decimal(0)
    baseline = card.baseline_sharpe
    stressed = card.stressed_cost_sharpe
    if baseline is None or stressed is None:
        return None
    if baseline <= 0:
        return Decimal(0)
    return _clamp(stressed / baseline)


def _regimes(card: Scorecard, policy: ScoringPolicy) -> Decimal | None:
    if card.regime_failed:
        return Decimal(0)
    sharpes = card.regime_sharpes
    if sharpes is None or len(sharpes) < 2:
        return None
    spread = max(sharpes) - min(sharpes)
    return _clamp(Decimal(1) - spread / policy.regime_spread_cap)


def _simplicity(card: Scorecard, policy: ScoringPolicy) -> Decimal | None:
    searched = card.specifications_searched
    if searched is None:
        return None
    if searched <= 1:
        return Decimal(1)
    if searched >= policy.simplicity_ceiling:
        return Decimal(0)
    span = policy.simplicity_ceiling - 1
    return Decimal(policy.simplicity_ceiling - searched) / Decimal(span)


def _metric(
    component: ScoreComponent,
    weights: dict[ScoreComponent, Decimal],
    raw: Decimal | None,
) -> ComponentScore:
    if raw is None:
        return ComponentScore(component, weights[component], ComponentStatus.MISSING, None)
    return ComponentScore(component, weights[component], ComponentStatus.SCORED, q(raw))


def _clamp(value: Decimal) -> Decimal:
    if value <= 0:
        return Decimal(0)
    if value >= 1:
        return Decimal(1)
    return value


def _require_card(card: Scorecard, policy: ScoringPolicy) -> None:
    _count(card.robustness_completed, policy.robustness_tests, "robustness count")
    _count(card.test_observations, None, "observation count")
    if card.specifications_searched is not None and (
        isinstance(card.specifications_searched, bool) or card.specifications_searched < 1
    ):
        raise ScoreRejected("scorecard is unreadable")
    _finite(card.oos_sharpe)
    _finite(card.baseline_sharpe)
    _finite(card.stressed_cost_sharpe)
    if card.regime_sharpes is not None:
        for value in card.regime_sharpes:
            _finite(value)


def _count(value: int | None, upper: int | None, label: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or value < 0 or (upper is not None and value > upper):
        raise ScoreRejected(f"{label} is outside the policy")


def _finite(value: Decimal | None) -> None:
    if value is None:
        return
    if not value.is_finite():
        raise ScoreRejected("scorecard is unreadable")


def _load(row: ResearchScoreRow) -> ResearchScore:
    policy = _policy_from_document(row.policy)
    if row.live_trade_authorized:
        raise ScoreRejected("a research score cannot authorize a live trade")
    return ResearchScore(
        id=row.id,
        experiment_id=row.experiment_id,
        policy_version=row.policy_version,
        purpose=policy.purpose,
        status=ScoreStatus(row.status),
        score=None if row.score is None else Decimal(row.score),
        components=_components_from_document(row.components, policy),
        weights=policy.weights,
        disclaimer=policy.disclaimer,
        live_trade_authorized=False,
        executed_at=row.executed_at,
    )


def _same(
    stored: ResearchScore,
    evaluation: ScoreEvaluation,
    stored_inputs: object,
    card: Scorecard,
) -> bool:
    if stored.status is not evaluation.status or stored.score != evaluation.score:
        return False
    if len(stored.components) != len(evaluation.components):
        return False
    for left, right in zip(stored.components, evaluation.components, strict=True):
        if left.component is not right.component or left.status is not right.status:
            return False
        if left.score != right.score or left.weight != right.weight:
            return False
    return _canonical(stored_inputs) == _canonical(_inputs_document(card))


def _policy_document(policy: ScoringPolicy) -> dict[str, object]:
    return {
        "version": policy.version,
        "purpose": policy.purpose,
        "weights": _weights_document(policy),
        "oos_sharpe_cap": str(policy.oos_sharpe_cap),
        "confidence_floor": policy.confidence_floor,
        "confidence_ceiling": policy.confidence_ceiling,
        "conditional_credit": str(policy.conditional_credit),
        "robustness_tests": policy.robustness_tests,
        "simplicity_ceiling": policy.simplicity_ceiling,
        "regime_spread_cap": str(policy.regime_spread_cap),
        "disclaimer": policy.disclaimer,
        "live_trade_authorized": False,
    }


def _policy_from_document(document: object) -> ScoringPolicy:
    body = _mapping(document)
    version = str(body.get("version", ""))
    registered = policy_for(version)
    if body.get("purpose") != registered.purpose:
        raise ScoreRejected("stored scoring policy does not match the registered policy")
    if body.get("live_trade_authorized") is not False:
        raise ScoreRejected("a research score cannot authorize a live trade")
    if _weights_document(registered) != body.get("weights"):
        raise ScoreRejected("stored scoring weights do not match the registered policy")
    return registered


def _weights_document(policy: ScoringPolicy) -> dict[str, str]:
    return {component.value: str(weight) for component, weight in policy.weights}


def _components_document(components: tuple[ComponentScore, ...]) -> list[dict[str, object]]:
    return [
        {
            "component": item.component.value,
            "weight": str(item.weight),
            "status": item.status.value,
            "score": None if item.score is None else str(item.score),
        }
        for item in components
    ]


def _components_from_document(
    document: object,
    policy: ScoringPolicy,
) -> tuple[ComponentScore, ...]:
    if not isinstance(document, list) or len(document) != len(policy.weights):
        raise ScoreRejected("stored component scores are incomplete")
    loaded: list[ComponentScore] = []
    for raw, (component, weight) in zip(document, policy.weights, strict=True):
        body = _mapping(raw)
        if body.get("component") != component.value or body.get("weight") != str(weight):
            raise ScoreRejected("stored component scores do not match the policy")
        score = body.get("score")
        loaded.append(
            ComponentScore(
                component=component,
                weight=weight,
                status=ComponentStatus(str(body.get("status"))),
                score=None if score is None else Decimal(str(score)),
            )
        )
    return tuple(loaded)


def _inputs_document(card: Scorecard) -> dict[str, object]:
    return {
        "oos_sharpe": _number(card.oos_sharpe),
        "robustness_completed": card.robustness_completed,
        "test_observations": card.test_observations,
        "mechanism": None if card.mechanism is None else card.mechanism.value,
        "baseline_sharpe": _number(card.baseline_sharpe),
        "stressed_cost_sharpe": _number(card.stressed_cost_sharpe),
        "transaction_cost_failed": card.transaction_cost_failed,
        "regime_sharpes": None
        if card.regime_sharpes is None
        else [_number(value) for value in card.regime_sharpes],
        "regime_failed": card.regime_failed,
        "capacity": None if card.capacity is None else card.capacity.value,
        "specifications_searched": card.specifications_searched,
    }


def _number(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    raise ScoreRejected("research score is unreadable")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ScoreRejected("timestamps must be timezone-aware")
