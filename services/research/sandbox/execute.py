"""Execute generated research code outside the laboratory process.

The child is denied network, shell, production files, and credential paths.
This module does not change host firewall, privilege, or filesystem policy.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

Json = dict[str, "Json"] | list["Json"] | str | int | float | bool | None

APPROVED_LIBRARIES: tuple[str, ...] = (
    "array",
    "bisect",
    "collections",
    "copy",
    "csv",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "numbers",
    "random",
    "statistics",
)

_WORKER = Path(__file__).with_name("worker.py")
_POLL_SECONDS = 0.02


class SandboxViolation(StrEnum):
    NETWORK = "NETWORK"
    CREDENTIALS = "CREDENTIALS"
    DATABASE = "DATABASE"
    FILESYSTEM = "FILESYSTEM"
    SHELL = "SHELL"
    TIMEOUT = "TIMEOUT"
    MEMORY = "MEMORY"
    OUTPUT = "OUTPUT"


@dataclass(frozen=True)
class ResourceLimits:
    timeout_seconds: float
    memory_bytes: int
    output_bytes: int

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.memory_bytes <= 0 or self.output_bytes <= 0:
            raise ValueError("resource limits must be positive")


@dataclass(frozen=True)
class SandboxFile:
    name: str
    content: bytes


@dataclass(frozen=True)
class SandboxExecution:
    experiment_id: UUID
    code_version: str
    dataset_version: str
    random_seed: int
    execution_timestamp: datetime
    metrics: dict[str, float]
    logs: tuple[str, ...]
    plots: tuple[SandboxFile, ...]
    artifacts: tuple[SandboxFile, ...]
    succeeded: bool
    violation: SandboxViolation | None


def execute_research_code(
    *,
    experiment_id: UUID,
    code_version: str,
    dataset_version: str,
    random_seed: int,
    execution_timestamp: datetime,
    dataset: Mapping[str, Json],
    specification: Mapping[str, Json],
    code: str,
    limits: ResourceLimits,
) -> SandboxExecution:
    """Run research code in a restricted child. The child result is untrusted."""
    _require_identity(code_version, dataset_version, execution_timestamp, code)
    work = Path(tempfile.mkdtemp(prefix="afil-sandbox-")).resolve()
    os.chmod(work, 0o700)
    try:
        _write_request(work, dataset, specification, code, random_seed, limits)
        (work / "plots").mkdir()
        (work / "artifacts").mkdir()
        process = _spawn(work)
        violation = _supervise(process, limits)
        return _record(
            work,
            process,
            violation,
            limits,
            experiment_id=experiment_id,
            code_version=code_version,
            dataset_version=dataset_version,
            random_seed=random_seed,
            execution_timestamp=execution_timestamp,
        )
    finally:
        _delete(work)


def _require_identity(
    code_version: str,
    dataset_version: str,
    execution_timestamp: datetime,
    code: str,
) -> None:
    if not code_version.strip() or not dataset_version.strip() or not code.strip():
        raise ValueError("code, code_version, and dataset_version are required")
    if execution_timestamp.tzinfo is None or execution_timestamp.utcoffset() is None:
        raise ValueError("execution_timestamp must be timezone-aware")


def _write_request(
    work: Path,
    dataset: Mapping[str, Json],
    specification: Mapping[str, Json],
    code: str,
    random_seed: int,
    limits: ResourceLimits,
) -> None:
    payload = {
        "dataset": dict(dataset),
        "specification": dict(specification),
        "code": code,
        "random_seed": random_seed,
        "timeout_seconds": limits.timeout_seconds,
        "output_bytes": limits.output_bytes,
        "approved_libraries": list(APPROVED_LIBRARIES),
    }
    (work / "request.json").write_text(json.dumps(payload), encoding="utf-8")


def _spawn(work: Path) -> subprocess.Popen[bytes]:
    pythons = _python_binaries()
    child_worker = work / "worker.py"
    shutil.copyfile(_WORKER, child_worker)
    worker = [pythons[0], "-I", str(child_worker), str(work)]
    sandbox = _sandbox_exec()
    command = worker if sandbox is None else [sandbox, "-p", _profile(work, pythons), *worker]
    return subprocess.Popen(
        command,
        cwd=work,
        env=_child_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


def _sandbox_exec() -> str | None:
    if sys.platform == "darwin":
        path = Path("/usr/bin/sandbox-exec")
        if not path.is_file():
            raise OSError("sandbox-exec is required on macOS and is not present")
        return str(path)
    return None


def _python_binaries() -> list[str]:
    base = Path(sys.base_prefix)
    candidates = (
        Path(sys.executable).resolve(),
        base / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}",
        base / "Resources/Python.app/Contents/MacOS/Python",
    )
    found = [str(path) for path in candidates if path.is_file()]
    if not found:
        raise OSError("python executable is not available for the sandbox")
    return found


def _profile(work: Path, pythons: list[str]) -> str:
    """Deny rules for the child only. Nothing here changes the host policy."""
    allows = "\n".join(f'(allow process-exec (literal "{path}"))' for path in pythons)
    home = str(Path.home().resolve())
    return f"""(version 1)
(allow default)
(deny network*)
{allows}
(deny process-exec)
(deny file-write* (require-not (subpath "{work}")))
(deny file-read* (subpath "/etc"))
(deny file-read* (subpath "{home}"))
(deny file-read* (subpath "/tmp"))
(deny file-read* (subpath "/private/tmp"))
(deny file-read* (subpath "/opt"))
(deny file-read* (subpath "/usr/local"))
"""


def _child_env() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        "PYTHONIOENCODING": "utf-8",
    }


def _supervise(process: subprocess.Popen[bytes], limits: ResourceLimits) -> SandboxViolation | None:
    assert process.stdout is not None
    assert process.stderr is not None
    tripped = threading.Event()
    threads = (
        _drain(process.stdout, limits.output_bytes, tripped),
        _drain(process.stderr, limits.output_bytes, tripped),
    )
    started = time.monotonic()
    reason: SandboxViolation | None = None
    while process.poll() is None:
        if tripped.is_set():
            reason = SandboxViolation.OUTPUT
            break
        if time.monotonic() - started > limits.timeout_seconds:
            reason = SandboxViolation.TIMEOUT
            break
        rss = _rss_bytes(process.pid)
        if rss is not None and rss > limits.memory_bytes:
            reason = SandboxViolation.MEMORY
            break
        time.sleep(_POLL_SECONDS)
    if reason is not None:
        _kill(process)
    for thread in threads:
        thread.join(timeout=1)
    return reason


def _drain(pipe: Any, limit: int, tripped: threading.Event) -> threading.Thread:
    def _read() -> None:
        total = 0
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                tripped.set()
                break
        pipe.close()

    thread = threading.Thread(target=_read, daemon=True)
    thread.start()
    return thread


def _rss_bytes(pid: int) -> int | None:
    probe = subprocess.run(
        ["/bin/ps", "-o", "rss=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = probe.stdout.strip()
    if not text.isdigit():
        return None
    return int(text) * 1024


def _kill(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _record(
    work: Path,
    process: subprocess.Popen[bytes],
    forced: SandboxViolation | None,
    limits: ResourceLimits,
    **identity: Any,
) -> SandboxExecution:
    payload: dict[str, Any] = {}
    result_path = work / "result.json"
    if result_path.is_file() and result_path.stat().st_size <= limits.output_bytes:
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
    violation = forced
    child_violation = payload.get("violation")
    if violation is None and isinstance(child_violation, str):
        violation = SandboxViolation(child_violation)
    succeeded = violation is None and payload.get("succeeded") is True and process.returncode == 0
    metrics = payload.get("metrics") if succeeded else {}
    if succeeded:
        logs = payload.get("logs")
    elif violation is not None:
        logs = [violation.value]
    else:
        logs = ["failed"]
    return SandboxExecution(
        experiment_id=identity["experiment_id"],
        code_version=identity["code_version"],
        dataset_version=identity["dataset_version"],
        random_seed=identity["random_seed"],
        execution_timestamp=identity["execution_timestamp"],
        metrics=_metrics(metrics),
        logs=_logs(logs),
        plots=_files(work / "plots", limits.output_bytes) if succeeded else (),
        artifacts=_files(work / "artifacts", limits.output_bytes) if succeeded else (),
        succeeded=succeeded,
        violation=violation,
    )


def _metrics(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    metrics: dict[str, float] = {}
    for name, metric in value.items():
        if (
            isinstance(name, str)
            and isinstance(metric, int | float)
            and not isinstance(metric, bool)
        ):
            metrics[name] = float(metric)
    return metrics


def _logs(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _files(directory: Path, output_bytes: int) -> tuple[SandboxFile, ...]:
    files: list[SandboxFile] = []
    if not directory.is_dir():
        return ()
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.stat().st_size <= output_bytes:
            files.append(SandboxFile(path.name, path.read_bytes()))
    return tuple(files)


def _delete(work: Path) -> None:
    for path in sorted(work.rglob("*"), reverse=True):
        if path.is_file() or path.is_symlink():
            path.unlink()
        else:
            path.rmdir()
    work.rmdir()
