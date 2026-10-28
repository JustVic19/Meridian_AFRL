"""Pydantic contracts for the AFIL API."""

from schemas.research import (
    ARTIFACT_TYPES,
    HYPOTHESIS_STATUSES,
    ArtifactType,
    CompanyRecord,
    DocumentRecord,
    ExperimentRecord,
    HypothesisRecord,
    HypothesisStatus,
    ObservationRecord,
    ResearchArtifactRecord,
    StrategyRecord,
)
from schemas.researcher import (
    CandidateHypothesis,
    KnownFact,
    ResearchFindings,
    ResearchRequest,
)
from schemas.system import HealthResponse, SystemStatusResponse

__all__ = [
    "ARTIFACT_TYPES",
    "HYPOTHESIS_STATUSES",
    "ArtifactType",
    "CandidateHypothesis",
    "CompanyRecord",
    "DocumentRecord",
    "KnownFact",
    "ResearchFindings",
    "ResearchRequest",
    "ExperimentRecord",
    "HealthResponse",
    "HypothesisRecord",
    "HypothesisStatus",
    "ObservationRecord",
    "ResearchArtifactRecord",
    "StrategyRecord",
    "SystemStatusResponse",
]
