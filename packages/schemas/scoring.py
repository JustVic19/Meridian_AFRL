"""Research-score names. The weights belong to the policy, not to a caller."""

from enum import StrEnum


class ScoreComponent(StrEnum):
    OOS_PERFORMANCE = "oos_performance"
    ROBUSTNESS = "robustness"
    STATISTICAL_CONFIDENCE = "statistical_confidence"
    ECONOMIC_MECHANISM = "economic_mechanism"
    TRANSACTION_COST_RESILIENCE = "transaction_cost_resilience"
    REGIME_STABILITY = "regime_stability"
    CAPACITY = "capacity"
    SIMPLICITY = "simplicity"


class ComponentStatus(StrEnum):
    SCORED = "SCORED"
    MISSING = "MISSING"


class ScoreStatus(StrEnum):
    SCORED = "SCORED"
    INCOMPLETE = "INCOMPLETE"


SCORE_COMPONENTS: tuple[str, ...] = tuple(component.value for component in ScoreComponent)
SCORE_STATUSES: tuple[str, ...] = tuple(status.value for status in ScoreStatus)
