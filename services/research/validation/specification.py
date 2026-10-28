"""Validate an experiment specification. This module does not run research code."""

from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

from schemas.experiment_spec import DatasetCatalog, ExperimentSpecification, SplitConfiguration
from schemas.hypothesis import EventDefinition

_VAGUE_BENCHMARKS = frozenset({"market", "the market", "stocks", "bad", "good"})
_VAGUE_EVENTS = frozenset({"event", "signal", "thing", "bad", "unknown", "move", "change"})


class SpecificationReason(StrEnum):
    MISSING_REPRODUCIBILITY = "MISSING_REPRODUCIBILITY"
    MISSING_BENCHMARK = "MISSING_BENCHMARK"
    MISSING_EXIT = "MISSING_EXIT"
    AMBIGUOUS_EVENT = "AMBIGUOUS_EVENT"
    LOOKAHEAD = "LOOKAHEAD"
    UNAVAILABLE_DATA = "UNAVAILABLE_DATA"


class SpecificationRejected(Exception):
    """The specification cannot be run. No code is generated."""

    def __init__(self, reasons: tuple[SpecificationReason, ...]) -> None:
        self.reasons = reasons
        super().__init__(", ".join(reason.value for reason in reasons))


@dataclass(frozen=True)
class DateWindow:
    start: date
    end: date


@dataclass(frozen=True)
class ValidatedSpecification:
    specification: ExperimentSpecification
    train: DateWindow
    validation: DateWindow
    test: DateWindow


def validate(
    specification: ExperimentSpecification,
    catalog: DatasetCatalog,
) -> ValidatedSpecification:
    """Accept a specification or reject it with stable reason codes."""
    reasons: list[SpecificationReason] = []
    if _missing_reproducibility(specification):
        reasons.append(SpecificationReason.MISSING_REPRODUCIBILITY)
    if _missing_benchmark(specification.benchmark):
        reasons.append(SpecificationReason.MISSING_BENCHMARK)
    if specification.exit is None:
        reasons.append(SpecificationReason.MISSING_EXIT)
    if _ambiguous_event(specification.event):
        reasons.append(SpecificationReason.AMBIGUOUS_EVENT)
    lookahead = _violates_point_in_time(specification, catalog)
    if lookahead:
        reasons.append(SpecificationReason.LOOKAHEAD)
    elif _unavailable(specification, catalog):
        reasons.append(SpecificationReason.UNAVAILABLE_DATA)
    if reasons:
        raise SpecificationRejected(tuple(reasons))
    train, validation, test = split_dataset(catalog.start, catalog.end, specification.split)
    return ValidatedSpecification(specification, train, validation, test)


def split_dataset(
    start: date,
    end: date,
    split: SplitConfiguration,
) -> tuple[DateWindow, DateWindow, DateWindow]:
    """Cut the inclusive dataset calendar into train, validation, and test."""
    count = (end - start).days + 1
    train_count = count * split.train_percent // 100
    validation_count = count * split.validation_percent // 100
    test_count = count - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise SpecificationRejected((SpecificationReason.UNAVAILABLE_DATA,))
    train_end = start + timedelta(days=train_count - 1)
    validation_start = train_end + timedelta(days=1)
    validation_end = validation_start + timedelta(days=validation_count - 1)
    test_start = validation_end + timedelta(days=1)
    return (
        DateWindow(start, train_end),
        DateWindow(validation_start, validation_end),
        DateWindow(test_start, end),
    )


def _missing_reproducibility(specification: ExperimentSpecification) -> bool:
    return (
        specification.random_seed is None
        or not specification.dataset_version.strip()
        or not specification.code_version.strip()
    )


def _missing_benchmark(benchmark: str) -> bool:
    return benchmark.strip().lower() in _VAGUE_BENCHMARKS or not benchmark.strip()


def _ambiguous_event(event: EventDefinition | None) -> bool:
    if event is None:
        return True
    label = event.type.strip().lower()
    return not label or label in _VAGUE_EVENTS


def _violates_point_in_time(
    specification: ExperimentSpecification,
    catalog: DatasetCatalog,
) -> bool:
    cutoff = specification.data_cutoff
    if cutoff is None or cutoff.tzinfo is None or cutoff.utcoffset() is None:
        return True
    if catalog.published_at > cutoff:
        return True
    return catalog.end > cutoff.date() or catalog.start > cutoff.date()


def _unavailable(specification: ExperimentSpecification, catalog: DatasetCatalog) -> bool:
    declared = specification.dataset_version.strip()
    if declared and declared != catalog.dataset_version:
        return True
    available = set(catalog.fields)
    required = set(specification.required_fields)
    for requirement in specification.data_requirements:
        required.update(requirement.fields)
    if not required <= available or any(not field for field in required):
        return True
    count = (catalog.end - catalog.start).days + 1
    split = specification.split
    train_count = count * split.train_percent // 100
    validation_count = count * split.validation_percent // 100
    test_count = count - train_count - validation_count
    return min(train_count, validation_count, test_count) < 1
