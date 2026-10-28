"""Sandbox tests. They do not use the experiment database or the QuantAgent."""

import os
import stat
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from research.sandbox import ResourceLimits, SandboxViolation, execute_research_code

EXPERIMENT_ID = UUID("00000000-0000-4000-8000-0000000000f1")
CODE_VERSION = "a1b2c3d"
DATASET_VERSION = "prices-2020-01"
SEED = 7
TIMESTAMP = datetime(2020, 1, 10, 21, tzinfo=UTC)
LIMITS = ResourceLimits(timeout_seconds=5, memory_bytes=256 * 1024 * 1024, output_bytes=100_000)
CODE = """
record_metric("count", len(dataset["close"]))
record_metric("mean", statistics.mean(dataset["close"]))
record_metric("draw", random.random())
log("counted")
save_plot("equity.txt", b"plot")
save_artifact("note.txt", "done")
"""


def _run(code: str, limits: ResourceLimits = LIMITS, **overrides: object) -> object:
    values: dict[str, object] = {
        "experiment_id": EXPERIMENT_ID,
        "code_version": CODE_VERSION,
        "dataset_version": DATASET_VERSION,
        "random_seed": SEED,
        "execution_timestamp": TIMESTAMP,
        "dataset": {"close": [1, 3, 5]},
        "specification": {"code_version": CODE_VERSION, "benchmark": "SECTOR"},
        "code": code,
        "limits": limits,
    }
    values.update(overrides)
    return execute_research_code(**values)  # type: ignore[arg-type]


def test_execution_returns_metrics_logs_plots_and_artifacts() -> None:
    first = _run(CODE)
    second = _run(CODE)
    assert first.succeeded and second.succeeded
    assert first.violation is None
    assert first.metrics["count"] == 3
    assert first.metrics["mean"] == 3
    assert first.metrics == second.metrics
    assert first.logs == ("counted",)
    assert first.plots == second.plots
    assert first.plots[0].name == "equity.txt"
    assert first.plots[0].content == b"plot"
    assert first.artifacts[0].content == b"done"
    assert first.experiment_id == EXPERIMENT_ID
    assert first.code_version == CODE_VERSION
    assert first.dataset_version == DATASET_VERSION
    assert first.random_seed == SEED
    assert first.execution_timestamp == TIMESTAMP
    assert first.execution_timestamp == second.execution_timestamp


def test_network_access_fails() -> None:
    result = _run('import socket\nsocket.create_connection(("example.com", 80), timeout=1)\n')
    assert result.succeeded is False
    assert result.violation is SandboxViolation.NETWORK
    assert result.metrics == {}


def test_broker_credentials_are_unavailable() -> None:
    os.environ["BROKER_API_KEY"] = "super-secret-broker-key"
    try:
        result = _run('import os\nos.environ["BROKER_API_KEY"]\n')
    finally:
        os.environ.pop("BROKER_API_KEY", None)
    rendered = repr(result)
    assert "super-secret-broker-key" not in rendered
    assert result.succeeded is False
    assert result.violation is SandboxViolation.CREDENTIALS


def test_database_write_fails() -> None:
    result = _run('import sqlite3\nsqlite3.connect(":memory:")\n')
    assert result.succeeded is False
    assert result.violation is SandboxViolation.DATABASE


def test_production_filesystem_access_fails(tmp_path: Path) -> None:
    sentinel = tmp_path / "production.txt"
    sentinel.write_text("keep")
    mode = sentinel.stat().st_mode
    result = _run(f"open({str(sentinel)!r}, 'w').write('changed')\n")
    assert result.succeeded is False
    assert result.violation is SandboxViolation.FILESYSTEM
    assert sentinel.read_text() == "keep"
    assert sentinel.stat().st_mode == mode
    credential = Path.home() / ".ssh"
    denied = _run(f"open({str(credential)!r})\n")
    assert denied.violation is SandboxViolation.CREDENTIALS


def test_shell_access_fails(tmp_path: Path) -> None:
    marker = tmp_path / "shell-breach"
    result = _run(f"import subprocess\nsubprocess.run(['/usr/bin/touch', {str(marker)!r}])\n")
    assert result.succeeded is False
    assert result.violation is SandboxViolation.SHELL
    assert marker.exists() is False
    assert stat.S_ISREG(Path("/usr/bin/touch").stat().st_mode)


def test_timeout_is_enforced() -> None:
    limits = ResourceLimits(
        timeout_seconds=0.4,
        memory_bytes=256 * 1024 * 1024,
        output_bytes=100_000,
    )
    result = _run("while True:\n    pass\n", limits)
    assert result.succeeded is False
    assert result.violation is SandboxViolation.TIMEOUT
    assert result.experiment_id == EXPERIMENT_ID
    assert result.code_version == CODE_VERSION


def test_memory_limit_is_enforced() -> None:
    limits = ResourceLimits(timeout_seconds=10, memory_bytes=96 * 1024 * 1024, output_bytes=100_000)
    code = "chunks = []\nwhile True:\n    chunks.append(bytearray(8 * 1024 * 1024))\n"
    result = _run(code, limits)
    assert result.succeeded is False
    assert result.violation is SandboxViolation.MEMORY


def test_output_limit_is_enforced() -> None:
    limits = ResourceLimits(timeout_seconds=5, memory_bytes=256 * 1024 * 1024, output_bytes=2048)
    result = _run('print("x" * 100000)\n', limits)
    assert result.succeeded is False
    assert result.violation is SandboxViolation.OUTPUT
    assert "x" * 100 not in "".join(result.logs)
