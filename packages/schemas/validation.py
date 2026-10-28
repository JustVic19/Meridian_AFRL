"""Validation policy names. The sequence is fixed; a caller does not choose tests."""

from enum import StrEnum


class ValidationTest(StrEnum):
    BASELINE = "BASELINE"
    TRANSACTION_COST = "TRANSACTION_COST"
    OUT_OF_SAMPLE = "OUT_OF_SAMPLE"
    ALTERNATIVE_THRESHOLD = "ALTERNATIVE_THRESHOLD"
    ALTERNATIVE_HOLDING_PERIOD = "ALTERNATIVE_HOLDING_PERIOD"
    SUBPERIOD = "SUBPERIOD"
    REGIME = "REGIME"
    SECTOR = "SECTOR"


class ValidationStatus(StrEnum):
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


VALIDATION_TEST_NAMES: tuple[str, ...] = tuple(test.value for test in ValidationTest)
VALIDATION_STATUSES: tuple[str, ...] = tuple(status.value for status in ValidationStatus)
