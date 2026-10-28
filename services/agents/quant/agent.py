"""QuantAgent. It writes research code after validation and never runs that code."""

import json
import re
from dataclasses import dataclass
from uuid import UUID, uuid5

from pydantic import ValidationError
from sqlalchemy.orm import Session

from agents.llm import LanguageModel, ModelRequest, ModelResponse
from api.models import Experiment, Hypothesis, ResearchArtifact
from prompts.quant import PROMPT_VERSION, SYSTEM_PROMPT, render_user_prompt
from research.validation.specification import ValidatedSpecification, validate
from schemas.experiment_spec import DatasetCatalog, ExperimentSpecification
from schemas.hypothesis import FormalizedHypothesis, HypothesisSpecification
from schemas.quant import ExecutionMetadata, QuantProgram
from schemas.research import ArtifactType

_NAMESPACE = UUID("d5e3f2bc-5ec0-4000-8000-0000000000c3")
_DEPENDENCY = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_DENIED_DEPENDENCIES = frozenset(
    {
        "aiohttp",
        "httpx",
        "psycopg",
        "psycopg2",
        "requests",
        "socket",
        "sqlalchemy",
        "subprocess",
        "urllib3",
    }
)
_PRODUCTION = re.compile(
    r"subprocess|os\.system|os\.popen|os\.remove|\bsocket\b|\brequests\b|"
    r"\burllib\b|\bhttpx\b|create_engine|\bpsycopg\b|\bsqlalchemy\b|"
    r"\beval\s*\(|\bexec\s*\(|__import__|\bopen\s*\(|\bshutil\b",
    re.IGNORECASE,
)
_CREDENTIAL = re.compile(
    r"os\.environ|\bgetenv\b|\bbroker\b|api[_-]?key|\bpassword\b|\bsecret\b|"
    r"\bcredential|DATABASE_URL",
    re.IGNORECASE,
)


class QuantRejected(Exception):
    """Code was not generated or was refused. Nothing was executed."""


@dataclass(frozen=True)
class SandboxBoundary:
    """Capabilities the QuantAgent does not have."""

    broker_credentials: None = None
    network: None = None
    production_database_write: None = None
    production_filesystem: None = None
    shell: None = None


@dataclass(frozen=True)
class _Draft:
    research_code: str
    dependencies: list[str]
    expected_outputs: list[str]


class QuantAgent:
    def __init__(self, session: Session, model: LanguageModel) -> None:
        self._session = session
        self._model = model
        self.boundary = SandboxBoundary()

    def generate(
        self,
        hypothesis: FormalizedHypothesis,
        specification: ExperimentSpecification,
        catalog: DatasetCatalog,
    ) -> QuantProgram:
        """Generate research code only after the specification validates."""
        validated = validate(specification, catalog)
        _require_stored_hypothesis(self._session, hypothesis)
        _require_same_design(validated.specification, hypothesis.specification)
        response = self._model.complete(
            ModelRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=render_user_prompt(
                    hypothesis.specification.model_dump_json(),
                    validated.specification.model_dump_json(),
                ),
                prompt_version=PROMPT_VERSION,
            )
        )
        draft = _draft_from_model(response)
        _reject_untrusted_code(draft)
        program = _program(hypothesis, validated, draft, response)
        _store(self._session, hypothesis, validated, catalog, draft, response, program)
        return program


def _require_stored_hypothesis(session: Session, hypothesis: FormalizedHypothesis) -> None:
    stored = session.get(Hypothesis, hypothesis.id)
    spec = hypothesis.specification
    if stored is None or stored.statement != spec.statement:
        raise QuantRejected("hypothesis is not stored")
    if stored.expected_horizon != spec.holding_period:
        raise QuantRejected("hypothesis is not stored")
    if stored.universe != spec.universe.model_dump(mode="json"):
        raise QuantRejected("hypothesis is not stored")
    if stored.entry_rule != spec.entry_rule.model_dump(mode="json"):
        raise QuantRejected("hypothesis is not stored")
    if stored.exit_rule != spec.exit_rule.model_dump(mode="json"):
        raise QuantRejected("hypothesis is not stored")


def _require_same_design(
    specification: ExperimentSpecification,
    hypothesis: HypothesisSpecification,
) -> None:
    if (
        specification.universe != hypothesis.universe
        or specification.event != hypothesis.event_definition
        or specification.entry != hypothesis.entry_rule
        or specification.exit != hypothesis.exit_rule
        or specification.benchmark != hypothesis.benchmark
    ):
        raise QuantRejected("specification does not match the hypothesis")


def _draft_from_model(response: ModelResponse) -> _Draft:
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise QuantRejected("model response is not json") from exc
    if not isinstance(payload, dict):
        raise QuantRejected("model response is not an object")
    for key in ("model_name", "model_version", "prompt_version", "code_version"):
        if key in payload:
            raise QuantRejected("model supplied version metadata")
    try:
        code = payload["research_code"]
        dependencies = payload["dependencies"]
        outputs = payload["expected_outputs"]
    except KeyError as exc:
        raise QuantRejected("model response is incomplete") from exc
    if not isinstance(code, str) or not isinstance(dependencies, list):
        raise QuantRejected("model response is incomplete")
    if not isinstance(outputs, list) or not all(isinstance(item, str) for item in outputs):
        raise QuantRejected("model response is incomplete")
    if not all(isinstance(item, str) for item in dependencies):
        raise QuantRejected("model response is incomplete")
    return _Draft(code, dependencies, outputs)


def _reject_untrusted_code(draft: _Draft) -> None:
    if not draft.research_code.strip():
        raise QuantRejected("research code is required")
    if _PRODUCTION.search(draft.research_code):
        raise QuantRejected("generated code requests production access")
    if _CREDENTIAL.search(draft.research_code):
        raise QuantRejected("generated code requests credential access")
    for name in draft.dependencies:
        if not _DEPENDENCY.fullmatch(name) or name.lower() in _DENIED_DEPENDENCIES:
            raise QuantRejected("generated code requests production access")


def _program(
    hypothesis: FormalizedHypothesis,
    validated: ValidatedSpecification,
    draft: _Draft,
    response: ModelResponse,
) -> QuantProgram:
    specification = validated.specification
    seed = specification.random_seed
    cutoff = specification.data_cutoff
    if seed is None or cutoff is None:
        raise QuantRejected("specification is not validated")
    metadata = ExecutionMetadata(
        dataset_version=specification.dataset_version,
        code_version=specification.code_version,
        random_seed=seed,
        data_cutoff=cutoff,
        train_start=validated.train.start,
        train_end=validated.train.end,
        validation_start=validated.validation.start,
        validation_end=validated.validation.end,
        test_start=validated.test.start,
        test_end=validated.test.end,
    )
    try:
        return QuantProgram(
            research_code=draft.research_code,
            execution_metadata=metadata,
            dependencies=draft.dependencies,
            expected_outputs=draft.expected_outputs,
            model_name=response.model_name,
            model_version=response.model_version,
            prompt_version=PROMPT_VERSION,
            experiment_id=_experiment_id(hypothesis, validated, draft, response),
            code_version=specification.code_version,
        )
    except ValidationError as exc:
        raise QuantRejected("model response is incomplete") from exc


def _experiment_id(
    hypothesis: FormalizedHypothesis,
    validated: ValidatedSpecification,
    draft: _Draft,
    response: ModelResponse,
) -> UUID:
    return uuid5(_NAMESPACE, "experiment:" + _canonical(hypothesis, validated, draft, response))


def _artifact_id(
    hypothesis: FormalizedHypothesis,
    validated: ValidatedSpecification,
    draft: _Draft,
    response: ModelResponse,
) -> UUID:
    return uuid5(_NAMESPACE, "artifact:" + _canonical(hypothesis, validated, draft, response))


def _canonical(
    hypothesis: FormalizedHypothesis,
    validated: ValidatedSpecification,
    draft: _Draft,
    response: ModelResponse,
) -> str:
    payload = {
        "hypothesis_id": str(hypothesis.id),
        "specification": validated.specification.model_dump(mode="json"),
        "train": [validated.train.start.isoformat(), validated.train.end.isoformat()],
        "validation": [
            validated.validation.start.isoformat(),
            validated.validation.end.isoformat(),
        ],
        "test": [validated.test.start.isoformat(), validated.test.end.isoformat()],
        "research_code": draft.research_code,
        "dependencies": draft.dependencies,
        "expected_outputs": draft.expected_outputs,
        "model_name": response.model_name,
        "model_version": response.model_version,
        "prompt_version": PROMPT_VERSION,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _store(
    session: Session,
    hypothesis: FormalizedHypothesis,
    validated: ValidatedSpecification,
    catalog: DatasetCatalog,
    draft: _Draft,
    response: ModelResponse,
    program: QuantProgram,
) -> None:
    if session.get(Experiment, program.experiment_id) is not None:
        return
    specification = validated.specification
    session.add(
        Experiment(
            id=program.experiment_id,
            hypothesis_id=hypothesis.id,
            dataset_version=specification.dataset_version,
            data_cutoff=specification.data_cutoff,
            code_version=program.code_version,
            train_start=validated.train.start,
            train_end=validated.train.end,
            validation_start=validated.validation.start,
            validation_end=validated.validation.end,
            test_start=validated.test.start,
            test_end=validated.test.end,
            specification={
                "specification": specification.model_dump(mode="json"),
                "catalog": catalog.model_dump(mode="json"),
            },
            status="CODE_GENERATED",
        )
    )
    session.flush()
    session.add(
        ResearchArtifact(
            id=_artifact_id(hypothesis, validated, draft, response),
            experiment_id=program.experiment_id,
            artifact_type=ArtifactType.CODE.value,
            content=program.model_dump_json(),
            model_name=program.model_name,
            model_version=program.model_version,
            prompt_version=program.prompt_version,
        )
    )
    session.flush()
