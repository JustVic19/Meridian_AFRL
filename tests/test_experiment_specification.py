"""Experiment specification validation. No code is generated or executed."""

from datetime import UTC, date, datetime

import pytest

from research.validation.specification import (
    SpecificationReason,
    SpecificationRejected,
    validate,
)
from schemas.experiment_spec import (
    DataRequirement,
    DatasetCatalog,
    ExperimentSpecification,
    SplitConfiguration,
)
from schemas.hypothesis import EntryRule, EventDefinition, ExitRule, UniverseDefinition

DATASET = "prices-2020-01"
CODE = "a1b2c3d"
START = date(2020, 1, 1)
END = date(2020, 1, 10)
CUTOFF = datetime(2020, 1, 10, 21, tzinfo=UTC)
PUBLISHED = datetime(2020, 1, 10, 18, tzinfo=UTC)


def _spec(**overrides: object) -> ExperimentSpecification:
    values: dict[str, object] = {
        "universe": UniverseDefinition(asset_class="US_EQUITY", min_market_cap=2_000_000_000),
        "event": EventDefinition(type="GUIDANCE_REVISION", threshold=-0.05),
        "entry": EntryRule(timing="NEXT_MARKET_OPEN"),
        "exit": ExitRule(holding_period_days=60),
        "benchmark": "SECTOR",
        "data_requirements": [DataRequirement(name="prices", fields=["close", "published_at"])],
        "required_fields": ["close", "published_at"],
        "transaction_cost_bps": 5,
        "slippage_bps": 2,
        "random_seed": 7,
        "dataset_version": DATASET,
        "code_version": CODE,
        "data_cutoff": CUTOFF,
    }
    values.update(overrides)
    return ExperimentSpecification(**values)  # type: ignore[arg-type]


def _catalog(**overrides: object) -> DatasetCatalog:
    values: dict[str, object] = {
        "dataset_version": DATASET,
        "start": START,
        "end": END,
        "fields": ["close", "published_at"],
        "published_at": PUBLISHED,
    }
    values.update(overrides)
    return DatasetCatalog(**values)  # type: ignore[arg-type]


def test_default_split_dates_come_from_the_dataset() -> None:
    spec = _spec()
    assert spec.split == SplitConfiguration()
    first = validate(spec, _catalog())
    second = validate(spec, _catalog())
    assert first == second
    assert first.train.start == date(2020, 1, 1)
    assert first.train.end == date(2020, 1, 6)
    assert first.validation.start == date(2020, 1, 7)
    assert first.validation.end == date(2020, 1, 8)
    assert first.test.start == date(2020, 1, 9)
    assert first.test.end == date(2020, 1, 10)
    assert first.specification.transaction_cost_bps == 5
    assert first.specification.slippage_bps == 2
    other_seed = validate(_spec(random_seed=99), _catalog())
    assert (other_seed.train, other_seed.validation, other_seed.test) == (
        first.train,
        first.validation,
        first.test,
    )


def test_unavailable_data_is_rejected() -> None:
    with pytest.raises(SpecificationRejected) as missing_field:
        validate(_spec(required_fields=["close", "volume"]), _catalog())
    assert missing_field.value.reasons == (SpecificationReason.UNAVAILABLE_DATA,)
    with pytest.raises(SpecificationRejected) as wrong_version:
        validate(_spec(dataset_version="prices-other"), _catalog())
    assert wrong_version.value.reasons == (SpecificationReason.UNAVAILABLE_DATA,)


def test_ambiguous_event_is_rejected() -> None:
    with pytest.raises(SpecificationRejected) as vague:
        validate(_spec(event=EventDefinition(type="event", threshold=-0.05)), _catalog())
    assert vague.value.reasons == (SpecificationReason.AMBIGUOUS_EVENT,)
    with pytest.raises(SpecificationRejected) as missing:
        validate(_spec(event=None), _catalog())
    assert missing.value.reasons == (SpecificationReason.AMBIGUOUS_EVENT,)


def test_missing_benchmark_and_exit_are_rejected() -> None:
    with pytest.raises(SpecificationRejected) as benchmark:
        validate(_spec(benchmark=""), _catalog())
    assert benchmark.value.reasons == (SpecificationReason.MISSING_BENCHMARK,)
    with pytest.raises(SpecificationRejected) as vague_benchmark:
        validate(_spec(benchmark="the market"), _catalog())
    assert vague_benchmark.value.reasons == (SpecificationReason.MISSING_BENCHMARK,)
    with pytest.raises(SpecificationRejected) as exit_rule:
        validate(_spec(exit=None), _catalog())
    assert exit_rule.value.reasons == (SpecificationReason.MISSING_EXIT,)


def test_point_in_time_violation_is_rejected() -> None:
    future_end = _catalog(end=date(2020, 1, 11))
    with pytest.raises(SpecificationRejected) as later_day:
        validate(_spec(), future_end)
    assert later_day.value.reasons == (SpecificationReason.LOOKAHEAD,)
    later_publication = _catalog(published_at=datetime(2020, 1, 10, 22, tzinfo=UTC))
    with pytest.raises(SpecificationRejected) as published:
        validate(_spec(), later_publication)
    assert published.value.reasons == (SpecificationReason.LOOKAHEAD,)
    with pytest.raises(SpecificationRejected) as missing_cutoff:
        validate(_spec(data_cutoff=None), _catalog())
    assert missing_cutoff.value.reasons == (SpecificationReason.LOOKAHEAD,)


def test_missing_reproducibility_metadata_is_rejected() -> None:
    with pytest.raises(SpecificationRejected) as seed:
        validate(_spec(random_seed=None), _catalog())
    assert seed.value.reasons == (SpecificationReason.MISSING_REPRODUCIBILITY,)
    with pytest.raises(SpecificationRejected) as code:
        validate(_spec(code_version=""), _catalog())
    assert code.value.reasons == (SpecificationReason.MISSING_REPRODUCIBILITY,)
    with pytest.raises(SpecificationRejected) as dataset:
        validate(_spec(dataset_version="  "), _catalog())
    assert dataset.value.reasons == (SpecificationReason.MISSING_REPRODUCIBILITY,)
