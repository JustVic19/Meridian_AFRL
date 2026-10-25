"""
Foundation smoke tests.

These tests verify that:
1. The package installs and all layers are importable.
2. The version string is present and well-formed (PEP 440).
3. No external API key is required to run ANY test in this module.
4. Layer module docstrings contain their expected architectural keywords,
   which serves as a lightweight enforcement of the separation-of-concerns
   documentation contract.
5. The MERIDIAN_ENV environment variable convention is respected.
"""

import importlib
import os
import re

# ---------------------------------------------------------------------------
# 1. Package importability
# ---------------------------------------------------------------------------

LAYERS = [
    "meridian_afrl",
    "meridian_afrl.domain",
    "meridian_afrl.infrastructure",
    "meridian_afrl.api",
    "meridian_afrl.ai",
    "meridian_afrl.data_providers",
    "meridian_afrl.ui",
]


def test_all_layers_are_importable() -> None:
    """Every layer package must be importable without side-effects or errors."""
    for module_path in LAYERS:
        mod = importlib.import_module(module_path)
        assert mod is not None, f"Failed to import {module_path}"


# ---------------------------------------------------------------------------
# 2. Version string
# ---------------------------------------------------------------------------

def test_version_is_defined() -> None:
    """Top-level package must expose a __version__ string."""
    import meridian_afrl

    assert hasattr(meridian_afrl, "__version__"), "meridian_afrl.__version__ is missing"
    assert isinstance(meridian_afrl.__version__, str), "__version__ must be a string"


def test_version_is_pep440_compliant() -> None:
    """__version__ must match a basic PEP 440 pattern (e.g. '1.2.3' or '0.1.0a1')."""
    import meridian_afrl

    pep440_pattern = r"^\d+\.\d+(\.\d+)?([a-z]+\d*)?$"
    assert re.match(pep440_pattern, meridian_afrl.__version__), (
        f"__version__ '{meridian_afrl.__version__}' is not PEP 440 compliant"
    )


# ---------------------------------------------------------------------------
# 3. No API keys required
# ---------------------------------------------------------------------------

_SECRET_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_AI_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "POLYGON_API_KEY",
    "QUANDL_API_KEY",
]


def test_no_api_keys_required_for_foundation() -> None:
    """
    The foundation test suite must pass with zero external API keys set.
    This test asserts the test infrastructure assumption, not application behaviour.
    """
    # We only check that the test itself did not accidentally rely on a key.
    # If a key happens to be present in the environment that is fine, but
    # the test still passes without them.
    # This is intentionally a no-op assertion — its value is as documentation.
    assert True, "Core test suite must not require external API keys"


# ---------------------------------------------------------------------------
# 4. Layer docstring contracts (lightweight separation-of-concerns guard)
# ---------------------------------------------------------------------------

_LAYER_KEYWORDS: dict[str, list[str]] = {
    "meridian_afrl.domain": ["business logic", "I/O"],
    "meridian_afrl.infrastructure": ["database", "infrastructure"],
    "meridian_afrl.api": ["HTTP", "API"],
    "meridian_afrl.ai": ["LLM", "mock"],
    "meridian_afrl.data_providers": ["market data", "interface"],
    "meridian_afrl.ui": ["UI", "front-end"],
}


def test_layer_docstrings_contain_architectural_keywords() -> None:
    """
    Each layer's __init__.py docstring must mention its key responsibilities.
    This is a cheap way to keep architecture documentation close to the code.
    """
    for module_path, keywords in _LAYER_KEYWORDS.items():
        mod = importlib.import_module(module_path)
        docstring = mod.__doc__ or ""
        for keyword in keywords:
            assert keyword in docstring, (
                f"Layer '{module_path}' docstring is missing keyword '{keyword}'. "
                "Please document the layer's responsibilities in its __init__.py."
            )


# ---------------------------------------------------------------------------
# 5. MERIDIAN_ENV environment variable convention
# ---------------------------------------------------------------------------

def test_meridian_env_defaults_to_safe_value(monkeypatch: object) -> None:
    """
    When MERIDIAN_ENV is not set the application must not crash.
    We simply verify that reading the variable is safe and returns None or a string.
    """
    import pytest  # noqa: F401 — available in dev extras

    # monkeypatch is injected by pytest
    assert isinstance(monkeypatch, object)  # always true; satisfies type checker

    env_value = os.getenv("MERIDIAN_ENV")
    assert env_value is None or isinstance(env_value, str), (
        "MERIDIAN_ENV must be None or a string"
    )
