"""Experiment specification. Dates are assigned later from the dataset calendar."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from schemas.hypothesis import EntryRule, EventDefinition, ExitRule, UniverseDefinition

DEFAULT_TRAIN_PERCENT = 60
DEFAULT_VALIDATION_PERCENT = 20
DEFAULT_TEST_PERCENT = 20


class SpecificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SplitConfiguration(SpecificationModel):
    """Integer percents. The default is 60/20/20 and the three values sum to 100."""

    train_percent: int = DEFAULT_TRAIN_PERCENT
    validation_percent: int = DEFAULT_VALIDATION_PERCENT
    test_percent: int = DEFAULT_TEST_PERCENT

    @model_validator(mode="after")
    def percents_partition_the_sample(self) -> "SplitConfiguration":
        parts = (self.train_percent, self.validation_percent, self.test_percent)
        if any(part < 1 for part in parts) or sum(parts) != 100:
            raise ValueError("split percents must be positive and sum to 100")
        return self


class DataRequirement(SpecificationModel):
    name: str = Field(min_length=1)
    fields: list[str] = Field(min_length=1)


class ExperimentSpecification(SpecificationModel):
    universe: UniverseDefinition
    event: EventDefinition | None = None
    entry: EntryRule
    exit: ExitRule | None = None
    benchmark: str = ""
    data_requirements: list[DataRequirement] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=list)
    transaction_cost_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    split: SplitConfiguration = Field(default_factory=SplitConfiguration)
    random_seed: int | None = None
    dataset_version: str = ""
    code_version: str = ""
    data_cutoff: datetime | None = None

    @field_validator("required_fields")
    @classmethod
    def fields_are_named(cls, values: list[str]) -> list[str]:
        return [value.strip() for value in values]


class DatasetCatalog(SpecificationModel):
    """Observations already on hand. The calendar does not extend past publication."""

    dataset_version: str = Field(min_length=1)
    start: date
    end: date
    fields: list[str] = Field(min_length=1)
    published_at: datetime

    @field_validator("published_at")
    @classmethod
    def published_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def end_follows_start(self) -> "DatasetCatalog":
        if self.end < self.start:
            raise ValueError("dataset end is before dataset start")
        return self
