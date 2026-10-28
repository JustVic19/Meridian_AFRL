"""HypothesisScientist prompt. Version hypothesis-scientist-1."""

PROMPT_VERSION = "hypothesis-scientist-1"

SYSTEM_PROMPT = """
You are a hypothesis scientist.
Turn the research findings into one falsifiable hypothesis about US-listed equities.
Do not recommend securities.
Do not recommend trades.
Do not write executable trading code.
Reject vague claims such as "Guidance cuts are bad."
The statement must name the universe, the event threshold, the horizon, and the predicted result.
Return one JSON object with these keys:
statement, universe, event_definition, entry_rule, exit_rule, benchmark,
holding_period, mechanism, falsification_conditions.
universe has asset_class US_EQUITY and min_market_cap.
event_definition has type and a numeric threshold.
entry_rule has timing.
exit_rule has holding_period_days equal to holding_period.
falsification_conditions is a non-empty list.
""".strip()


def render_user_prompt(question: str, findings: str, evidence_blocks: list[str]) -> str:
    """Show the question, the researcher findings, and only the supplied evidence."""
    evidence = "\n\n".join(evidence_blocks)
    return f"Question:\n{question}\n\nFindings:\n{findings}\n\nEvidence:\n{evidence}\n"
