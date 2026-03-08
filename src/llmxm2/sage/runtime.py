from __future__ import annotations

import json
import math
import subprocess
import time
from typing import Any, Mapping

from llmxm2.sage.types import ExecutionResult, SageRuntimeConfig

SAGE_RUNNER_CODE = r'''
import contextlib
import io
import json
import os
import sys

from sage.all import *


def _emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=True))


def _main():
    if len(sys.argv) > 1:
        raw_payload = sys.argv[1]
    elif os.environ.get("LLMXM2_PAYLOAD"):
        raw_payload = os.environ["LLMXM2_PAYLOAD"]
    else:
        raw_payload = sys.stdin.read()

    payload = json.loads(raw_payload)
    mode = payload.get("mode", "code")

    if mode == "code":
        code = payload.get("code", "")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code must be a non-empty string")

        result_var = payload.get("result_var", "RESULT")
        if not isinstance(result_var, str) or not result_var.strip():
            result_var = "RESULT"

        namespace = {"__builtins__": __builtins__}
        stdout_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, namespace, namespace)

        result_obj = namespace.get(result_var)
        if result_obj is None and "result" in namespace:
            result_obj = namespace.get("result")

        _emit(
            {
                "status": "ok",
                "result_plain": "" if result_obj is None else str(result_obj),
                "result_latex": "" if result_obj is None else str(latex(result_obj)),
                "stdout": stdout_buffer.getvalue(),
            }
        )
        return

    if mode == "operation":
        operation = payload.get("operation")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")

        args = payload.get("args", [])
        kwargs = payload.get("kwargs", {})
        if not isinstance(args, list):
            raise ValueError("args must be a list")
        if not isinstance(kwargs, dict):
            raise ValueError("kwargs must be an object")

        target = globals().get(operation)
        if not callable(target):
            raise ValueError(f"Sage callable not found: {operation}")

        result = target(*args, **kwargs)
        _emit(
            {
                "status": "ok",
                "result_plain": str(result),
                "result_latex": str(latex(result)),
                "stdout": "",
            }
        )
        return

    raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    try:
        _main()
    except Exception as exc:
        _emit(
            {
                "status": "error",
                "result_plain": "",
                "result_latex": "",
                "stdout": "",
                "error": str(exc),
            }
        )
'''


class SageRuntime:
    def __init__(self, config: SageRuntimeConfig):
        if not config.image:
            raise ValueError("Sage runtime requires a Docker image.")
        self.config = config

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SageRuntime":
        cfg_dict = dict(cfg)
        return cls(
            SageRuntimeConfig(
                image=str(cfg_dict.get("image", "")),
                platform=str(cfg_dict.get("platform", "")),
                entrypoint=str(cfg_dict.get("entrypoint", "/bin/bash")),
                cpus=float(cfg_dict.get("cpus", 1.0)),
                memory=str(cfg_dict.get("memory", "1g")),
                pids_limit=int(cfg_dict.get("pids_limit", 128)),
                wall_timeout_sec=float(cfg_dict.get("wall_timeout_sec", 8.0)),
                cpu_limit_sec=float(cfg_dict.get("cpu_limit_sec", 5.0)),
                output_max_bytes=int(cfg_dict.get("output_max_bytes", 262_144)),
                user=str(cfg_dict.get("user", "")),
                home_dir=str(cfg_dict.get("home_dir", "/tmp")),
                dot_sage_dir=str(cfg_dict.get("dot_sage_dir", "/tmp/.sage")),
                progress_logs=bool(cfg_dict.get("progress_logs", False)),
            )
        )

    def execute_sage_code(
        self,
        code: str,
        timeout_sec: float | None = None,
        result_var: str = "RESULT",
    ) -> ExecutionResult:
        payload = {
            "mode": "code",
            "code": code,
            "result_var": result_var,
        }
        return self._execute(payload=payload, timeout_sec=timeout_sec)

    def execute_operation(
        self,
        operation: str,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
        timeout_sec: float | None = None,
    ) -> ExecutionResult:
        payload = {
            "mode": "operation",
            "operation": operation,
            "args": list(args or []),
            "kwargs": dict(kwargs or {}),
        }
        return self._execute(payload=payload, timeout_sec=timeout_sec)

    def _execute(self, payload: dict[str, Any], timeout_sec: float | None = None) -> ExecutionResult:
        payload_json = json.dumps(payload, ensure_ascii=True)
        if len(payload_json.encode("utf-8")) > self.config.output_max_bytes:
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                runtime_ms=0,
                stdout="",
                stderr="",
                error="Payload exceeds output_max_bytes.",
            )

        cmd = self._build_docker_cmd(payload_json)
        timeout = timeout_sec if timeout_sec is not None else self.config.wall_timeout_sec

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                cmd,
                input=payload_json,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            runtime_ms = int((time.perf_counter() - started) * 1000)
            return ExecutionResult(
                status="timeout",
                result_plain="",
                result_latex="",
                runtime_ms=runtime_ms,
                stdout="",
                stderr="",
                error="Execution timed out.",
            )
        except OSError as exc:
            runtime_ms = int((time.perf_counter() - started) * 1000)
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                runtime_ms=runtime_ms,
                stdout="",
                stderr="",
                error=f"Failed to invoke docker: {exc}",
            )

        runtime_ms = int((time.perf_counter() - started) * 1000)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        if len(stdout.encode("utf-8")) > self.config.output_max_bytes:
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                runtime_ms=runtime_ms,
                stdout="",
                stderr=stderr,
                error="Output exceeds output_max_bytes.",
                exit_code=completed.returncode,
            )

        parsed = self._parse_runner_output(stdout)
        if parsed is None:
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                runtime_ms=runtime_ms,
                stdout=stdout,
                stderr=stderr,
                error="Runner output was not valid JSON.",
                exit_code=completed.returncode,
            )

        status = str(parsed.get("status", "error"))
        error = str(parsed.get("error", ""))

        if completed.returncode != 0 and status == "ok":
            status = "error"
            if not error:
                error = stderr.strip() or f"Exit code {completed.returncode}"

        return ExecutionResult(
            status=status,
            result_plain=str(parsed.get("result_plain", "")),
            result_latex=str(parsed.get("result_latex", "")),
            runtime_ms=runtime_ms,
            stdout=str(parsed.get("stdout", "")),
            stderr=stderr,
            error=error,
            exit_code=completed.returncode,
        )

    def _build_docker_cmd(self, payload_json: str) -> list[str]:
        _ = payload_json
        cmd = ["docker", "run", "--rm"]

        if self.config.entrypoint:
            cmd.extend(["--entrypoint", self.config.entrypoint])
        if self.config.platform:
            cmd.extend(["--platform", self.config.platform])
        if self.config.user:
            cmd.extend(["--user", self.config.user])

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

        if self.config.cpu_limit_sec > 0:
            cmd.extend(["--ulimit", f"cpu={max(1, math.ceil(self.config.cpu_limit_sec))}"])

        cmd.append(self.config.image)
        cmd.extend(self._runtime_exec_args(payload_json))
        return cmd

    def _runtime_exec_args(self, payload_json: str) -> list[str]:
        entry = self.config.entrypoint.rsplit("/", maxsplit=1)[-1].lower()
        if entry in {"bash", "sh"}:
            shell_flag = "-lc" if entry == "bash" else "-c"
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
    def _parse_runner_output(stdout: str) -> dict[str, Any] | None:
        text = stdout.strip()
        if not text:
            return None
        candidate = text.splitlines()[-1]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None


def execute_sage_code(runtime: SageRuntime, code: str, timeout_sec: float | None = None) -> ExecutionResult:
    return runtime.execute_sage_code(code=code, timeout_sec=timeout_sec)
