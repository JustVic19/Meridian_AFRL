"""HypothesisScientist. Each formulation is a new hypothesis row and a stored artifact."""

import json
import re
from uuid import UUID, uuid5

from pydantic import ValidationError
from sqlalchemy.orm import Session

from agents.llm import LanguageModel, ModelRequest, ModelResponse
from api.models import Hypothesis, ResearchArtifact
from prompts.hypothesis import PROMPT_VERSION, SYSTEM_PROMPT, render_user_prompt
from schemas.hypothesis import (
    FormalizedHypothesis,
    HypothesisRequest,
    HypothesisSpecification,
)
from schemas.research import ArtifactType, HypothesisStatus

_NAMESPACE = UUID("c4d2e1ab-5ec0-4000-8000-0000000000b2")
_TRADE = re.compile(
    r"\b(?:buy|sell|short)\s+[A-Z]{1,5}\b"
    r"|\brecommend(?:s|ed|ing)?\b(?:\s+\w+){0,6}\s+"
    r"(?:buy|sell|short|trade|security|securities|position)s?\b",
    re.IGNORECASE,
)
_CODE_FENCE = re.compile(r"```|\bdef\s+[A-Za-z_]\w*\s*\(")
_CODE_IMPORT = re.compile(r"^\s*import\s+\w+", re.MULTILINE)


class InvalidHypothesis(Exception):
    """The model did not return a falsifiable hypothesis."""


class MalformedHypothesis(InvalidHypothesis):
    """The model contradicted the recorded model or prompt version."""


class HypothesisScientist:
    def __init__(self, session: Session, model: LanguageModel) -> None:
        self._session = session
        self._model = model

    def formalize(self, request: HypothesisRequest) -> FormalizedHypothesis:
        """Store a new hypothesis. An existing row is never updated."""
        version = _version(self._session, request.parent_hypothesis_id)
        response = self._model.complete(
            ModelRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=render_user_prompt(
                    request.question,
                    request.findings.model_dump_json(),
                    _blocks(request),
                ),
                prompt_version=PROMPT_VERSION,
            )
        )
        specification = _specification_from_model(response)
        _reject_trade_language(specification)
        formalized = FormalizedHypothesis(
            id=_hypothesis_id(request, specification),
            version=version,
            parent_hypothesis_id=request.parent_hypothesis_id,
            specification=specification,
        )
        _store(self._session, request, formalized)
        return formalized


def _version(session: Session, parent_id: UUID | None) -> int:
    version = 1
    seen: set[UUID] = set()
    current = parent_id
    while current is not None:
        if current in seen:
            raise InvalidHypothesis("hypothesis lineage is cyclic")
        seen.add(current)
        parent = session.get(Hypothesis, current)
        if parent is None:
            raise InvalidHypothesis("parent hypothesis does not exist")
        version += 1
        current = parent.parent_hypothesis_id
    return version


def _blocks(request: HypothesisRequest) -> list[str]:
    return [
        "\n".join(
            (
                f"document_id={item.document_id}",
                f"accession_number={item.accession_number or ''}",
                item.excerpt,
            )
        )
        for item in request.evidence
    ]


def _specification_from_model(response: ModelResponse) -> HypothesisSpecification:
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise InvalidHypothesis("model response is not json") from exc
    if not isinstance(payload, dict):
        raise InvalidHypothesis("model response is not an object")
    supplied = {
        "model_name": response.model_name,
        "model_version": response.model_version,
        "prompt_version": PROMPT_VERSION,
    }
    for key, value in supplied.items():
        if key in payload and payload[key] != value:
            raise MalformedHypothesis(key)
    payload.update(supplied)
    try:
        return HypothesisSpecification.model_validate(payload)
    except ValidationError as exc:
        message = str(exc)
        if "vague hypothesis" in message:
            raise InvalidHypothesis("vague hypothesis") from exc
        if "event_definition" in message or "threshold" in message:
            raise InvalidHypothesis("missing event definition") from exc
        if "falsification" in message:
            raise InvalidHypothesis("missing falsification conditions") from exc
        raise InvalidHypothesis("hypothesis is not falsifiable") from exc


def _reject_trade_language(specification: HypothesisSpecification) -> None:
    text = "\n".join(
        [
            specification.statement,
            specification.mechanism,
            specification.benchmark,
            *specification.falsification_conditions,
        ]
    )
    if _TRADE.search(text) or _CODE_FENCE.search(text) or _CODE_IMPORT.search(text):
        raise InvalidHypothesis("hypothesis recommends a trade or includes code")


def _hypothesis_id(request: HypothesisRequest, specification: HypothesisSpecification) -> UUID:
    return uuid5(_NAMESPACE, "hypothesis:" + _canonical(request, specification))


def _artifact_id(request: HypothesisRequest, specification: HypothesisSpecification) -> UUID:
    return uuid5(_NAMESPACE, "artifact:" + _canonical(request, specification))


def _canonical(request: HypothesisRequest, specification: HypothesisSpecification) -> str:
    payload = {
        "input": request.model_dump(mode="json"),
        "output": specification.model_dump(mode="json"),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _store(session: Session, request: HypothesisRequest, formalized: FormalizedHypothesis) -> None:
    if session.get(Hypothesis, formalized.id) is not None:
        return
    specification = formalized.specification
    session.add(
        Hypothesis(
            id=formalized.id,
            title=_title(specification),
            statement=specification.statement,
            universe=specification.universe.model_dump(mode="json"),
            mechanism=specification.mechanism,
            entry_rule=specification.entry_rule.model_dump(mode="json"),
            exit_rule=specification.exit_rule.model_dump(mode="json"),
            expected_horizon=specification.holding_period,
            parent_hypothesis_id=formalized.parent_hypothesis_id,
            status=HypothesisStatus.DRAFT.value,
        )
    )
    session.add(
        ResearchArtifact(
            id=_artifact_id(request, specification),
            experiment_id=None,
            artifact_type=ArtifactType.HYPOTHESIS.value,
            content=_canonical(request, specification),
            model_name=specification.model_name,
            model_version=specification.model_version,
            prompt_version=specification.prompt_version,
        )
    )
    session.flush()


def _title(specification: HypothesisSpecification) -> str:
    return f"{specification.event_definition.type} {specification.holding_period}d"
