"""Child entry point for generated research code. It is started with Python -I."""

from __future__ import annotations

import errno
import json
import resource
import sys
from pathlib import Path

_DENIED = {
    "asyncio": "NETWORK",
    "ctypes": "SHELL",
    "dbm": "DATABASE",
    "ftplib": "NETWORK",
    "http": "NETWORK",
    "httpx": "NETWORK",
    "importlib": "SHELL",
    "multiprocessing": "SHELL",
    "pickle": "SHELL",
    "psycopg": "DATABASE",
    "psycopg2": "DATABASE",
    "pty": "SHELL",
    "requests": "NETWORK",
    "runpy": "SHELL",
    "shelve": "DATABASE",
    "smtplib": "NETWORK",
    "socket": "NETWORK",
    "sqlite3": "DATABASE",
    "sqlalchemy": "DATABASE",
    "ssl": "NETWORK",
    "subprocess": "SHELL",
    "urllib": "NETWORK",
    "webbrowser": "NETWORK",
}
_CREDENTIAL_MARKERS = (
    ".aws",
    ".env",
    ".netrc",
    ".pgpass",
    ".ssh",
    "broker",
    "credential",
    "id_rsa",
    "secret",
)
_SENSITIVE_ENV = ("BROKER", "CREDENTIAL", "DATABASE", "KEY", "PASSWORD", "SECRET", "TOKEN")
_SHELL_OS = {
    "execl",
    "execle",
    "execlp",
    "execv",
    "execve",
    "execvp",
    "fork",
    "kill",
    "popen",
    "spawnl",
    "spawnv",
    "system",
}


class SandboxFault(BaseException):
    def __init__(self, violation: str) -> None:
        self.violation = violation
        super().__init__(violation)


def main() -> None:
    work = Path(sys.argv[1]).resolve()
    request = json.loads((work / "request.json").read_text(encoding="utf-8"))
    _limit_process(int(request["output_bytes"]), float(request["timeout_seconds"]))
    outcome = _run(work, request)
    (work / "result.json").write_text(json.dumps(outcome), encoding="utf-8")


def _limit_process(output_bytes: int, timeout_seconds: float) -> None:
    _soft_limit(resource.RLIMIT_FSIZE, output_bytes)
    _soft_limit(resource.RLIMIT_CPU, max(2, int(timeout_seconds) + 1))
    _soft_limit(resource.RLIMIT_NPROC, 0)


def _soft_limit(kind: int, value: int) -> None:
    _soft, hard = resource.getrlimit(kind)
    resource.setrlimit(kind, (value, hard))


def _run(work: Path, request: dict[str, object]) -> dict[str, object]:
    logs: list[str] = []
    metrics: dict[str, float] = {}
    try:
        namespace = _namespace(work, request, logs, metrics)
        code = request["code"]
        if not isinstance(code, str):
            raise SandboxFault("FILESYSTEM")
        exec(compile(code, "<research>", "exec"), namespace, namespace)
        return {"succeeded": True, "violation": None, "metrics": metrics, "logs": logs}
    except SandboxFault as exc:
        return {
            "succeeded": False,
            "violation": exc.violation,
            "metrics": {},
            "logs": [exc.violation],
        }
    except OSError as exc:
        violation = "OUTPUT" if exc.errno == errno.EFBIG else "FILESYSTEM"
        return {"succeeded": False, "violation": violation, "metrics": {}, "logs": [violation]}
    except Exception:
        return {"succeeded": False, "violation": None, "metrics": {}, "logs": ["failed"]}


def _namespace(
    work: Path,
    request: dict[str, object],
    logs: list[str],
    metrics: dict[str, float],
) -> dict[str, object]:
    approved = request["approved_libraries"]
    if not isinstance(approved, list):
        raise SandboxFault("FILESYSTEM")
    allowed = {name for name in approved if isinstance(name, str)}
    output_bytes = _as_int(request["output_bytes"])
    used = 0
    real_import = __import__

    def _import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        del globals, locals, fromlist
        if level != 0:
            raise SandboxFault("SHELL")
        root = name.split(".", 1)[0]
        if root in _DENIED:
            raise SandboxFault(_DENIED[root])
        if root == "os":
            return _restricted_os()
        if root not in allowed:
            raise SandboxFault("FILESYSTEM")
        return real_import(name)

    def _account(size: int) -> None:
        nonlocal used
        used += size
        if used > output_bytes:
            raise SandboxFault("OUTPUT")

    def _log(message: object) -> None:
        text = str(message)
        _account(len(text.encode()))
        logs.append(text[:1000])

    def _metric(name: object, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise SandboxFault("OUTPUT")
        if not isinstance(name, str):
            raise SandboxFault("OUTPUT")
        if value != value or value in {float("inf"), float("-inf")}:
            raise SandboxFault("OUTPUT")
        metrics[name] = float(value)

    def _save(directory: str, name: object, content: object) -> None:
        filename = _filename(name)
        data = content.encode() if isinstance(content, str) else content
        if not isinstance(data, bytes):
            raise SandboxFault("OUTPUT")
        _account(len(data))
        (work / directory / filename).write_bytes(data)

    modules = {name: real_import(name) for name in sorted(allowed)}
    random_module = modules.get("random")
    if random_module is not None:
        random_module.seed(_as_int(request["random_seed"]))
    builtins = {
        "__import__": _import,
        "Exception": Exception,
        "False": False,
        "None": None,
        "RuntimeError": RuntimeError,
        "True": True,
        "TypeError": TypeError,
        "ValueError": ValueError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "bytearray": bytearray,
        "bytes": bytes,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "open": _open,
        "print": _log,
        "range": range,
        "repr": repr,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }
    namespace: dict[str, object] = {
        "__builtins__": builtins,
        "dataset": request["dataset"],
        "log": _log,
        "random_seed": request["random_seed"],
        "record_metric": _metric,
        "save_artifact": lambda name, content: _save("artifacts", name, content),
        "save_plot": lambda name, content: _save("plots", name, content),
        "specification": request["specification"],
    }
    namespace.update(modules)
    return namespace


def _restricted_os() -> object:
    class _Environ(dict[str, str]):
        def __getitem__(self, key: str) -> str:
            del key
            raise SandboxFault("CREDENTIALS")

    class _Os:
        environ = _Environ()

        def getenv(self, key: str, default: str | None = None) -> str | None:
            del default
            raise SandboxFault("CREDENTIALS")

        def __getattr__(self, name: str) -> object:
            if name in _SHELL_OS or name.startswith(("exec", "spawn")):
                raise SandboxFault("SHELL")
            raise SandboxFault("FILESYSTEM")

    return _Os()


def _open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
    del mode, args, kwargs
    path = str(file).lower()
    if any(marker in path for marker in _CREDENTIAL_MARKERS):
        raise SandboxFault("CREDENTIALS")
    if path.endswith((".db", ".sqlite", ".sqlite3")):
        raise SandboxFault("DATABASE")
    raise SandboxFault("FILESYSTEM")


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SandboxFault("FILESYSTEM")
    return value


def _filename(name: object) -> str:
    if not isinstance(name, str) or "/" in name or "\\" in name or name in {"", ".", ".."}:
        raise SandboxFault("FILESYSTEM")
    return name


if __name__ == "__main__":
    main()
