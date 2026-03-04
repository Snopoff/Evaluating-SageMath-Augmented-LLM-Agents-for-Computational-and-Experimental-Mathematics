from __future__ import annotations

import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from .policy import (
    ERROR_BUDGET,
    ERROR_EXEC,
    ERROR_TIMEOUT,
    SageEvalRequest,
)

_DIGEST_PATTERN = re.compile(r"@sha256:[a-fA-F0-9]{64}$")


SAGE_RUNNER_CODE = r'''
import contextlib
import io
import json
import os
import sys

from sage.all import *


def _preview_namespace(namespace):
    preview = {}
    for name, value in namespace.items():
        if not isinstance(name, str):
            continue
        if name.startswith("_"):
            continue
        if name == "__builtins__":
            continue
        try:
            preview[name] = str(value)
        except Exception:
            preview[name] = f"<unrepr:{type(value).__name__}>"
        if len(preview) >= 32:
            break
    return preview


def _sage_snippet(args):
    code = args.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("sage_snippet requires non-empty code string")

    result_var = args.get("result_var", "RESULT")
    if not isinstance(result_var, str) or not result_var.strip():
        result_var = "RESULT"
    include_locals = bool(args.get("include_locals", False))

    stdout_buffer = io.StringIO()
    sandbox_ns = {"__builtins__": __builtins__}
    with contextlib.redirect_stdout(stdout_buffer):
        exec(code, sandbox_ns, sandbox_ns)

    result_obj = None
    if result_var in sandbox_ns:
        result_obj = sandbox_ns.get(result_var)
    elif "result" in sandbox_ns:
        result_obj = sandbox_ns.get("result")

    payload = {
        "result_var": result_var,
        "result_repr": "" if result_obj is None else str(result_obj),
        "stdout": stdout_buffer.getvalue(),
    }
    if include_locals:
        payload["locals_preview"] = _preview_namespace(sandbox_ns)
    return payload


def _coerce_generic_value(value, coerce_symbolic_strings):
    if isinstance(value, str) and coerce_symbolic_strings:
        try:
            return SR(value)
        except Exception:
            return value
    if isinstance(value, list):
        return [_coerce_generic_value(item, coerce_symbolic_strings) for item in value]
    if isinstance(value, dict):
        return {str(k): _coerce_generic_value(v, coerce_symbolic_strings) for (k, v) in value.items()}
    return value


def _invoke_generic_operation(operation, args):
    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be a non-empty string")
    if operation.startswith("_"):
        raise ValueError("operation cannot target private symbols")
    if operation in {"open", "eval", "exec", "compile", "__import__"}:
        raise ValueError("operation is not permitted")

    positional_args = args.get("positional_args", args.get("args", []))
    keyword_args = args.get("keyword_args", args.get("kwargs", {}))
    coerce_symbolic_strings = bool(args.get("coerce_symbolic_strings", False))

    if not isinstance(positional_args, list):
        raise ValueError("generic operation requires positional_args list")
    if not isinstance(keyword_args, dict):
        raise ValueError("generic operation requires keyword_args object")

    target = globals().get(operation)
    if not callable(target):
        raise ValueError(f"Sage callable not found: {operation}")

    call_args = [_coerce_generic_value(v, coerce_symbolic_strings) for v in positional_args]
    call_kwargs = {str(k): _coerce_generic_value(v, coerce_symbolic_strings) for (k, v) in keyword_args.items()}
    return target(*call_args, **call_kwargs)


def _emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))


def _main():
    raw_payload = ""
    if len(sys.argv) > 1 and isinstance(sys.argv[1], str) and sys.argv[1]:
        raw_payload = sys.argv[1]
    elif os.environ.get("LLMXM2_PAYLOAD"):
        raw_payload = os.environ["LLMXM2_PAYLOAD"]
    else:
        raw_payload = sys.stdin.read()
    payload = json.loads(raw_payload)
    operation = payload["operation"]
    args = payload.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("tool args must be an object")
    if operation == "sage_snippet":
        result = _sage_snippet(args)
    else:
        result = _invoke_generic_operation(operation, args)

    _emit(
        {
            "status": "ok",
            "result_plain": str(result),
            "result_latex": str(latex(result)),
            "error_code": "NONE",
        }
    )


if __name__ == "__main__":
    try:
        _main()
    except Exception as exc:
        _emit(
            {
                "status": "error",
                "result_plain": "",
                "result_latex": "",
                "error_code": "EXEC_ERROR",
                "message": str(exc),
            }
        )
'''


@dataclass(frozen=True)
class DockerRuntimeConfig:
    image: str
    platform: str = ""
    entrypoint: str = "/bin/bash"
    allowed_registries: list[str] = field(default_factory=lambda: ["docker.io", "ghcr.io"])
    cpus: float = 1.0
    memory: str = "1g"
    pids_limit: int = 128
    wall_seconds: float = 8.0
    cpu_seconds: float = 5.0
    output_max_bytes: int = 262_144
    user: str = ""
    home_dir: str = "/tmp"
    dot_sage_dir: str = "/tmp/.sage"
    progress_logs: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    result_plain: str
    result_latex: str
    runtime_ms: int
    error_code: str
    message: str = ""


class DockerSageExecutor:
    """Runs Sage operations inside a locked-down Docker container."""

    def __init__(self, config: DockerRuntimeConfig):
        self.config = config
        self._validate_image_reference()

    def _progress(self, message: str) -> None:
        if self.config.progress_logs:
            print(f"[progress][docker-exec] {message}", flush=True)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "DockerSageExecutor":
        user = cls._normalize_user(str(cfg.get("user", "")))
        entrypoint = cls._normalize_entrypoint(str(cfg.get("entrypoint", "/bin/bash")))
        return cls(
            DockerRuntimeConfig(
                image=str(cfg.get("image")),
                platform=str(cfg.get("platform", "")),
                entrypoint=entrypoint,
                allowed_registries=list(cfg.get("allowed_registries", ["docker.io", "ghcr.io"])),
                cpus=float(cfg.get("cpus", 1.0)),
                memory=str(cfg.get("memory", "1g")),
                pids_limit=int(cfg.get("pids_limit", 128)),
                wall_seconds=float(cfg.get("wall_seconds", 8.0)),
                cpu_seconds=float(cfg.get("cpu_seconds", 5.0)),
                output_max_bytes=int(cfg.get("output_max_bytes", 262_144)),
                user=user,
                home_dir=str(cfg.get("home_dir", "/tmp")),
                dot_sage_dir=str(cfg.get("dot_sage_dir", "/tmp/.sage")),
                progress_logs=bool(cfg.get("progress_logs", False)),
            )
        )

    def execute(self, request: SageEvalRequest) -> ExecutionResult:
        payload = {
            "operation": request.operation,
            "args": request.args,
            "assumptions": request.assumptions,
            "request_id": request.request_id,
            "budget_profile": request.budget_profile,
        }
        payload_json = json.dumps(payload, ensure_ascii=True)

        if len(payload_json.encode("utf-8")) > self.config.output_max_bytes:
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                runtime_ms=0,
                error_code=ERROR_BUDGET,
                message="Request payload exceeds max output size.",
            )

        start = time.perf_counter()
        completed: subprocess.CompletedProcess[str] | None = None

        try:
            entrypoint_attempts = self._entrypoint_attempts()
            user_attempts = self._user_attempts()

            for user_index, user in enumerate(user_attempts):
                continue_to_next_user = False
                for entry_index, entry in enumerate(entrypoint_attempts):
                    self._progress(
                        "starting docker execution "
                        f"(operation={request.operation}, wall_timeout={self.config.wall_seconds}s, "
                        f"entrypoint={entry or '<default-entrypoint>'}, user={user or '<image-default>'})"
                    )
                    cmd = self._build_docker_cmd(
                        entrypoint_override=entry,
                        user_override=user,
                        payload_json=payload_json,
                    )
                    completed = subprocess.run(
                        cmd,
                        input=payload_json,
                        capture_output=True,
                        text=True,
                        timeout=self.config.wall_seconds,
                        check=False,
                    )
                    if completed.returncode == 0:
                        continue_to_next_user = False
                        break

                    stderr = completed.stderr or ""
                    if not self._is_retryable_entrypoint_error(stderr):
                        continue_to_next_user = False
                        break

                    if entry_index + 1 < len(entrypoint_attempts):
                        next_entrypoint = entrypoint_attempts[entry_index + 1] or "<default-entrypoint>"
                        self._progress(
                            "entrypoint failed before Sage runner started; "
                            f"retrying with {next_entrypoint}"
                        )
                    else:
                        continue_to_next_user = user_index + 1 < len(user_attempts)
                        if continue_to_next_user:
                            self._progress("retrying with image default user")

                if completed is not None and completed.returncode == 0:
                    break
                if not continue_to_next_user:
                    break
        except subprocess.TimeoutExpired:
            runtime_ms = int((time.perf_counter() - start) * 1000)
            self._progress(f"docker execution timed out after {runtime_ms}ms")
            return ExecutionResult(
                status="timeout",
                result_plain="",
                result_latex="",
                runtime_ms=runtime_ms,
                error_code=ERROR_TIMEOUT,
                message="Docker execution timed out.",
            )
        except OSError as exc:
            runtime_ms = int((time.perf_counter() - start) * 1000)
            self._progress(f"docker invocation failed after {runtime_ms}ms: {exc}")
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                runtime_ms=runtime_ms,
                error_code=ERROR_EXEC,
                message=f"Docker invocation failed: {exc}",
            )

        if completed is None:
            runtime_ms = int((time.perf_counter() - start) * 1000)
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                runtime_ms=runtime_ms,
                error_code=ERROR_EXEC,
                message="Docker execution did not start.",
            )

        runtime_ms = int((time.perf_counter() - start) * 1000)
        self._progress(f"docker execution completed (returncode={completed.returncode}, runtime_ms={runtime_ms})")
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if completed.returncode != 0 and stderr.strip():
            self._progress(f"docker stderr: {stderr.strip()[:400]}")

        if len(stdout.encode("utf-8")) > self.config.output_max_bytes:
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                runtime_ms=runtime_ms,
                error_code=ERROR_BUDGET,
                message="Tool output exceeds max output size.",
            )

        parsed = self._parse_runner_output(stdout)
        if parsed is None:
            message = f"Invalid runner output. stderr={stderr.strip()[:500]}"
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                runtime_ms=runtime_ms,
                error_code=ERROR_EXEC,
                message=message,
            )

        status = str(parsed.get("status", "error"))
        result_plain = str(parsed.get("result_plain", ""))
        result_latex = str(parsed.get("result_latex", ""))
        error_code = str(parsed.get("error_code", ERROR_EXEC))

        if completed.returncode != 0 and status == "ok":
            status = "error"
            error_code = ERROR_EXEC

        if status == "timeout":
            error_code = ERROR_TIMEOUT

        message = str(parsed.get("message", ""))
        if stderr.strip() and not message:
            message = stderr.strip()[:500]

        return ExecutionResult(
            status=status,
            result_plain=result_plain,
            result_latex=result_latex,
            runtime_ms=runtime_ms,
            error_code=error_code,
            message=message,
        )

    def _build_docker_cmd(
        self,
        entrypoint_override: str | None = None,
        user_override: str | None = None,
        payload_json: str = "",
    ) -> list[str]:
        entrypoint = entrypoint_override if entrypoint_override is not None else self.config.entrypoint
        user = user_override if user_override is not None else self.config.user
        cmd = [
            "docker",
            "run",
            "--rm",
        ]
        if entrypoint:
            cmd.extend(["--entrypoint", entrypoint])
        if self.config.platform:
            cmd.extend(["--platform", self.config.platform])
        cmd.extend(["-e", f"HOME={self.config.home_dir}"])
        cmd.extend(["-e", f"DOT_SAGE={self.config.dot_sage_dir}"])
        cmd.extend(
            [
                "--network",
                "none",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,nodev,size=256m",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                str(self.config.pids_limit),
                "--memory",
                self.config.memory,
                "--cpus",
                str(self.config.cpus),
            ]
        )
        if self.config.cpu_seconds > 0:
            cmd.extend(["--ulimit", f"cpu={max(1, math.ceil(self.config.cpu_seconds))}"])
        if user:
            cmd.extend(["--user", user])
        cmd.append(self.config.image)
        cmd.extend(self._runtime_exec_args(entrypoint, payload_json))
        return cmd

    def _runtime_exec_args(self, entrypoint: str, payload_json: str) -> list[str]:
        entrypoint_name = entrypoint.rsplit("/", maxsplit=1)[-1].lower()
        if entrypoint_name in {"bash", "sh"}:
            shell_flag = "-lc" if entrypoint_name == "bash" else "-c"
            shell_script = (
                'SAGE_BIN="sage"; '
                'if ! command -v "$SAGE_BIN" >/dev/null 2>&1; then '
                'for p in /home/sage/sage/sage /opt/sagemath/sage /usr/local/bin/sage /usr/bin/sage; do '
                '[ -x "$p" ] && SAGE_BIN="$p" && break; '
                "done; "
                "fi; "
                'if ! command -v "$SAGE_BIN" >/dev/null 2>&1 && [ ! -x "$SAGE_BIN" ]; then '
                'echo "sage executable not found in container" >&2; '
                "exit 127; "
                "fi; "
                'exec "$SAGE_BIN" -python -c "$1" "$2"'
            )
            return [shell_flag, shell_script, "_", SAGE_RUNNER_CODE, payload_json]

        return ["-python", "-c", SAGE_RUNNER_CODE, payload_json]

    @staticmethod
    def _is_retryable_entrypoint_error(stderr: str) -> bool:
        text = stderr.lower()
        return (
            'exec: "sage": executable file not found in $path' in text
            or "sage executable not found in container" in text
            or "exec: sage: not found" in text
            or "sage: not found" in text
            or 'exec: "/bin/bash": stat /bin/bash: no such file or directory' in text
            or 'exec: "bash": executable file not found in $path' in text
            or 'exec: "/bin/sh": stat /bin/sh: no such file or directory' in text
            or 'exec: "sh": executable file not found in $path' in text
        )

    def _entrypoint_attempts(self) -> list[str]:
        attempts: list[str] = []
        configured = self._normalize_entrypoint(self.config.entrypoint)
        attempts.append(configured)

        for fallback in ("/bin/bash", "/bin/sh", ""):
            if fallback not in attempts:
                attempts.append(fallback)
        return attempts

    def _user_attempts(self) -> list[str]:
        attempts: list[str] = []
        configured = self._normalize_user(self.config.user)
        attempts.append(configured)
        if configured:
            attempts.append("")
        return attempts

    @staticmethod
    def _normalize_entrypoint(entrypoint: str) -> str:
        value = (entrypoint or "").strip()
        if not value:
            return "/bin/bash"
        if value == "sage":
            return "/bin/bash"
        return value

    @staticmethod
    def _normalize_user(user: str) -> str:
        value = (user or "").strip()
        if value in {"", "auto", "image-default", "default"}:
            return ""
        return value

    def _validate_image_reference(self) -> None:
        image = self.config.image
        if not image or image == "None":
            raise ValueError("Docker image must be configured for Sage executor.")

        if not _DIGEST_PATTERN.search(image):
            raise ValueError("Sage image must be pinned by digest (…@sha256:<64hex>).")

        registry = self._extract_registry(image)
        if registry not in set(self.config.allowed_registries):
            raise ValueError(f"Docker registry '{registry}' is not allowlisted.")

    @staticmethod
    def _extract_registry(image: str) -> str:
        first = image.split("/", 1)[0]
        if "." in first or ":" in first or first == "localhost":
            return first
        return "docker.io"

    @staticmethod
    def _parse_runner_output(stdout: str) -> dict[str, Any] | None:
        text = stdout.strip()
        if not text:
            return None

        candidate = text.splitlines()[-1]
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
            return None
        except json.JSONDecodeError:
            return None
