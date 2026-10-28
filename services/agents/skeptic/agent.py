"""Skeptic agent. It tries to falsify a recorded result and does not assign the grade."""

import json
import re
from datetime import date, datetime
from uuid import UUID, uuid5

from pydantic import ValidationError
from sqlalchemy.orm import Session

from agents.llm import LanguageModel, ModelRequest, ModelResponse
from agents.skeptic.checks import Assessment, grade_checks, verdict_for
from api.models import Experiment, Hypothesis, ResearchArtifact
from prompts.skeptic import PROMPT_VERSION, SYSTEM_PROMPT, render_user_prompt
from schemas.research import ArtifactType
from schemas.skeptic import (
    AuditFacts,
    CheckFinding,
    ExperimentSnapshot,
    ModelNotes,
    SkepticReport,
)

_ARTIFACT_NAMESPACE = UUID("f9c7d6f0-5ec0-4000-8000-0000000000e7")
_DEFENSE = re.compile(
    r"\bhypothesis is (?:supported|confirmed|true|correct|robust)\b"
    r"|\b(?:supports|confirms|validates|proves) the hypothesis\b"
    r"|\bwe accept the hypothesis\b"
    r"|\bthis check passes\b"
    r"|\bgrade should be pass\b",
    re.IGNORECASE,
)
_ATTEMPT = re.compile(r"\b(?:falsif\w*|objection)\b", re.IGNORECASE)
_TRADE = re.compile(
    r"\b(?:buy|sell|short)\s+[A-Z]{1,5}\b|\brecommend(?:s|ed|ing)?\b(?:\s+\w+){0,6}\s+"
    r"(?:buy|sell|short|trade|security|securities|position)s?\b",
    re.IGNORECASE,
)
_CODE = re.compile(r"```|\bdef\s+[A-Za-z_]\w*\s*\(|^\s*import\s+\w+", re.MULTILINE)


class SkepticRejected(Exception):
    """The review would defend, leak, or grade a result that is not recorded."""


class MalformedModelResponse(SkepticRejected):
    """The model response is not the skeptic's notes object."""


class UnsupportedEvidence(SkepticRejected):
    """A note cites evidence the check did not record."""


class HypothesisDefense(SkepticRejected):
    """The model defended the hypothesis instead of challenging it."""


class SkepticAgent:
    def __init__(self, session: Session, model: LanguageModel) -> None:
        self._session = session
        self._model = model

    def review(self, experiment_id: UUID, audit: AuditFacts) -> SkepticReport:
        """Grade the stored result, then keep only a challenge that cites that evidence."""
        snapshot = _snapshot(self._session, experiment_id)
        findings = grade_checks(experiment_id, snapshot, audit)
        response = self._model.complete(
            ModelRequest(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=render_user_prompt(str(experiment_id), _lines(findings)),
                prompt_version=PROMPT_VERSION,
            )
        )
        notes = _notes(response)
        report = _report(experiment_id, findings, notes, response)
        _store(self._session, experiment_id, audit, findings, report)
        return report


def _snapshot(session: Session, experiment_id: UUID) -> ExperimentSnapshot:
    row = session.get(Experiment, experiment_id)
    if row is None:
        raise SkepticRejected("experiment does not exist")
    if row.status != "TEST_RECORDED" or row.completed_at is None:
        raise SkepticRejected("test results are not recorded")
    hypothesis = session.get(Hypothesis, row.hypothesis_id)
    if hypothesis is None:
        raise SkepticRejected("hypothesis does not exist")
    return ExperimentSnapshot(
        data_cutoff=row.data_cutoff,
        train_start=_date(row.train_start),
        train_end=_date(row.train_end),
        test_start=_date(row.test_start),
        test_end=_date(row.test_end),
        specification=_mapping(row.specification),
        parameters=_mapping(row.parameters),
        statistical_tests=_mapping(row.statistical_tests),
        hypothesis_statement=hypothesis.statement,
        hypothesis_mechanism=hypothesis.mechanism,
    )


def _notes(response: ModelResponse) -> ModelNotes:
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise MalformedModelResponse("model response is not json") from exc
    if not isinstance(payload, dict):
        raise MalformedModelResponse("model response is not an object")
    try:
        return ModelNotes.model_validate(payload)
    except ValidationError as exc:
        raise MalformedModelResponse("model response does not match the schema") from exc


def _report(
    experiment_id: UUID,
    findings: list[Assessment],
    notes: ModelNotes,
    response: ModelResponse,
) -> SkepticReport:
    by_check = {item.check: item for item in findings}
    supplied = {note.check: note for note in notes.notes}
    if set(supplied) != set(by_check) or len(notes.notes) != len(findings):
        raise MalformedModelResponse("notes must cover every methodological check once")
    checks: list[CheckFinding] = []
    for finding in findings:
        note = supplied[finding.check]
        expected = [item.id for item in finding.evidence]
        if note.evidence_ids != expected:
            raise UnsupportedEvidence("note cites evidence that was not recorded")
        _reject_defense(note.challenge)
        checks.append(
            CheckFinding(
                check=finding.check,
                grade=finding.grade,
                reason=finding.reason,
                evidence_ids=expected,
                challenge=note.challenge,
            )
        )
    return SkepticReport(
        experiment_id=experiment_id,
        verdict=verdict_for([item.grade for item in findings]),
        checks=checks,
        model_name=response.model_name,
        model_version=response.model_version,
        prompt_version=PROMPT_VERSION,
    )


def _reject_defense(challenge: str) -> None:
    if _DEFENSE.search(challenge) or _TRADE.search(challenge) or _CODE.search(challenge):
        raise HypothesisDefense("skeptic output defends the hypothesis")
    if _ATTEMPT.search(challenge) is None:
        raise HypothesisDefense("challenge does not attempt to falsify the result")


def _store(
    session: Session,
    experiment_id: UUID,
    audit: AuditFacts,
    findings: list[Assessment],
    report: SkepticReport,
) -> None:
    payload = {
        "input": {"experiment_id": str(experiment_id), "audit": audit.model_dump(mode="json")},
        "evidence": [
            item.model_dump(mode="json") for finding in findings for item in finding.evidence
        ],
        "output": report.model_dump(mode="json"),
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    artifact_id = uuid5(_ARTIFACT_NAMESPACE, content)
    if session.get(ResearchArtifact, artifact_id) is not None:
        return
    session.add(
        ResearchArtifact(
            id=artifact_id,
            experiment_id=experiment_id,
            artifact_type=ArtifactType.SKEPTIC_REPORT.value,
            content=content,
            model_name=report.model_name,
            model_version=report.model_version,
            prompt_version=report.prompt_version,
        )
    )
    session.flush()


def _lines(findings: list[Assessment]) -> list[str]:
    lines: list[str] = []
    for finding in findings:
        lines.append(f"check={finding.check.value} grade={finding.grade.value}")
        lines.append(f"reason={finding.reason}")
        for evidence in finding.evidence:
            lines.append(
                f"evidence_id={evidence.id} field={evidence.field} observed={evidence.observed}"
            )
    return lines


def _date(value: date | datetime | None) -> date:
    if not isinstance(value, date) or isinstance(value, datetime):
        raise SkepticRejected("experiment period is missing")
    return value


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    raise SkepticRejected("experiment record is unreadable")
