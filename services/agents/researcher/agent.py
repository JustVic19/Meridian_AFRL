"""Researcher agent. It records findings and does not recommend trades."""

import json
import re
from uuid import UUID, uuid5

from pydantic import ValidationError
from sqlalchemy.orm import Session

from agents.llm import LanguageModel, ModelRequest, ModelResponse
from agents.researcher.evidence import EvidenceExcerpt, EvidenceSource
from api.models import ResearchArtifact
from prompts.researcher import PROMPT_VERSION, SYSTEM_PROMPT, render_user_prompt
from schemas.research import ArtifactType
from schemas.researcher import KnownFact, ResearchFindings, ResearchRequest

_ARTIFACT_NAMESPACE = UUID("b7e1c0de-5ec0-4000-8000-0000000000a1")
_TRADE = re.compile(
    r"\b(?:buy|sell|short)\s+[A-Z]{1,5}\b"
    r"|\brecommend(?:s|ed|ing)?\b(?:\s+\w+){0,6}\s+"
    r"(?:buy|sell|short|trade|security|securities|position)s?\b",
    re.IGNORECASE,
)
_CODE_FENCE = re.compile(r"```|\bdef\s+[A-Za-z_]\w*\s*\(")
_CODE_IMPORT = re.compile(r"^\s*import\s+\w+", re.MULTILINE)


class MalformedModelResponse(Exception):
    """The model response is not the researcher's JSON object."""


class UnsupportedClaim(Exception):
    """A known fact cites evidence the retrieval step did not return."""


class TradeRecommendation(Exception):
    """The model recommended a trade, a security, or executable code."""


class Researcher:
    def __init__(self, session: Session, evidence: EvidenceSource, model: LanguageModel) -> None:
        self._session = session
        self._evidence = evidence
        self._model = model

    def investigate(self, request: ResearchRequest) -> ResearchFindings:
        """Retrieve evidence, check the model output, and store one artifact."""
        if not request.question.strip():
            raise ValueError("question is required")
        evidence = self._evidence.collect(request)
        response = self._model.complete(
            ModelRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=render_user_prompt(request.question, _blocks(evidence)),
                prompt_version=PROMPT_VERSION,
            )
        )
        findings = _findings_from_model(response)
        _require_supported(findings, evidence)
        _reject_trade_language(findings)
        _store(self._session, request, evidence, findings)
        return findings


def _blocks(evidence: tuple[EvidenceExcerpt, ...]) -> list[str]:
    return [
        "\n".join(
            (
                f"document_id={item.document_id}",
                f"accession_number={item.accession_number or ''}",
                f"published_at={item.published_at.isoformat()}",
                item.excerpt,
            )
        )
        for item in evidence
    ]


def _findings_from_model(response: ModelResponse) -> ResearchFindings:
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise MalformedModelResponse("model response is not json") from exc
    if not isinstance(payload, dict):
        raise MalformedModelResponse("model response is not an object")
    supplied = {
        "model_name": response.model_name,
        "model_version": response.model_version,
        "prompt_version": PROMPT_VERSION,
    }
    for key, value in supplied.items():
        if key in payload and payload[key] != value:
            raise MalformedModelResponse(f"{key} does not match the provider")
    payload.update(supplied)
    try:
        return ResearchFindings.model_validate(payload)
    except ValidationError as exc:
        raise MalformedModelResponse("model response does not match the schema") from exc


def _require_supported(findings: ResearchFindings, evidence: tuple[EvidenceExcerpt, ...]) -> None:
    retrieved = {item.document_id for item in evidence}
    if not retrieved and findings.known_facts:
        raise UnsupportedClaim("known facts require retrieved evidence")
    for fact in findings.known_facts:
        _require_known_ids(fact, retrieved)


def _require_known_ids(fact: KnownFact, retrieved: set[UUID]) -> None:
    if any(evidence_id not in retrieved for evidence_id in fact.evidence_ids):
        raise UnsupportedClaim("known fact cites evidence that was not retrieved")


def _reject_trade_language(findings: ResearchFindings) -> None:
    text = "\n".join(_statements(findings))
    if _TRADE.search(text) or _CODE_FENCE.search(text) or _CODE_IMPORT.search(text):
        raise TradeRecommendation("model output recommends a trade or includes code")


def _statements(findings: ResearchFindings) -> list[str]:
    statements = [
        findings.research_summary,
        *findings.unknowns,
        *findings.mechanisms,
        *findings.potential_variables,
        *findings.falsification_conditions,
    ]
    statements.extend(fact.statement for fact in findings.known_facts)
    statements.extend(item.statement for item in findings.candidate_hypotheses)
    return statements


def _store(
    session: Session,
    request: ResearchRequest,
    evidence: tuple[EvidenceExcerpt, ...],
    findings: ResearchFindings,
) -> None:
    payload = _payload(request, evidence, findings)
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    artifact_id = uuid5(_ARTIFACT_NAMESPACE, content)
    if session.get(ResearchArtifact, artifact_id) is not None:
        return
    session.add(
        ResearchArtifact(
            id=artifact_id,
            experiment_id=None,
            artifact_type=ArtifactType.RESEARCH_NOTE.value,
            content=content,
            model_name=findings.model_name,
            model_version=findings.model_version,
            prompt_version=findings.prompt_version,
        )
    )
    session.flush()


def _payload(
    request: ResearchRequest,
    evidence: tuple[EvidenceExcerpt, ...],
    findings: ResearchFindings,
) -> dict[str, object]:
    return {
        "input": request.model_dump(mode="json"),
        "evidence": [
            {
                "document_id": str(item.document_id),
                "accession_number": item.accession_number,
                "published_at": item.published_at.isoformat(),
                "excerpt": item.excerpt,
            }
            for item in evidence
        ],
        "output": findings.model_dump(mode="json"),
    }
