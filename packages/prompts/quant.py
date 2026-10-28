"""Quant prompt. Version quant-1. Generated code is untrusted and is not executed."""

PROMPT_VERSION = "quant-1"

SYSTEM_PROMPT = """
You write research code for one already validated US-equity experiment.
Return one JSON object with these keys: research_code, dependencies, expected_outputs.
research_code is a Python function that reads only the fields named in the specification.
dependencies is a list of package names.
expected_outputs is a non-empty list of result names.
Do not use the network.
Do not read or write credentials.
Do not open a shell.
Do not write to a database or filesystem.
Do not recommend a trade.
The code will be stored and will not be executed by you.
""".strip()


def render_user_prompt(hypothesis: str, specification: str) -> str:
    """Show the validated hypothesis and the validated specification. Nothing else."""
    return f"Hypothesis:\n{hypothesis}\n\nSpecification:\n{specification}\n"
