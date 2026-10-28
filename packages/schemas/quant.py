"""Quant output. The code is stored as text and is not executed."""

from datetime import date, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuantModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExecutionMetadata(QuantModel):
    dataset_version: str = Field(min_length=1)
    code_version: str = Field(min_length=1)
    random_seed: int
    data_cutoff: datetime
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date


class QuantProgram(QuantModel):
    research_code: str = Field(min_length=1)
    execution_metadata: ExecutionMetadata
    dependencies: list[str]
    expected_outputs: list[str] = Field(min_length=1)
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    experiment_id: UUID
    code_version: str = Field(min_length=1)

    @field_validator("dependencies", "expected_outputs")
    @classmethod
    def names_are_present(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("blank name")
        return values
