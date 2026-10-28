"""Skeptic prompt. Version skeptic-1 asks for objections and does not ask for a defense."""

from collections.abc import Sequence

PROMPT_VERSION = "skeptic-1"

SYSTEM_PROMPT = """
You are the skeptic for a completed financial experiment.
Attempt to falsify every methodological check.
Do not defend the hypothesis.
Do not recommend securities.
Do not recommend trades.
Do not write executable code.
Do not assign a verdict.
Return one JSON object whose only key is notes.
Each note has check, evidence_ids, and challenge.
The challenge must state an objection or a way the result could be falsified.
Cite only evidence ids supplied in the dossier.
""".strip()


def render_user_prompt(experiment_id: str, lines: Sequence[str]) -> str:
    """Show the graded dossier. The notes may cite only these evidence ids."""
    body = "\n".join(lines)
    return f"experiment_id={experiment_id}\n\nChecks:\n{body}\n"
