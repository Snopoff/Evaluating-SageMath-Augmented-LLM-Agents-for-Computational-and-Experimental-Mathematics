import json
import math
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from src.sage.types import ExecutionResult, SageRuntimeConfig
from src.utils.console_logging import ConsoleLogger

CONTAINER_SOURCE_FILE = "/tmp/llmxcas_sage_exec.py"

SAGE_RUNNER_CODE = r"""
import contextlib
import io
import json
import sys
import traceback

from sage.repl.preparse import preparse_file
from sage.all import *


def emit(payload):
    sys.stdout.write("LLM_SAGE_RESULT:")
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def to_result_data(value, depth=0):
    if depth > 6:
        return str(value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        return {str(key): to_result_data(item, depth + 1) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_result_data(item, depth + 1) for item in value]

    return str(value)


def main():
    if len(sys.argv) != 2:
        raise ValueError("expected JSON payload as the first argument")

    payload = json.loads(sys.argv[1])
    source_file = payload.get("source_file", "")
    if not isinstance(source_file, str):
        source_file = ""
    result_var = payload.get("result_var", "RESULT")
    if not isinstance(result_var, str) or not result_var.strip():
        result_var = "RESULT"

    if source_file:
        with open(source_file, "r", encoding="utf-8") as handle:
            code = handle.read()
    else:
        code = payload.get("code", "")

    if not isinstance(code, str) or not code.strip():
        raise ValueError("code must be a non-empty string")

    # Copy runner globals so user code can access Sage symbols.
    namespace = dict(globals())
    namespace["__builtins__"] = __builtins__
    if source_file:
        namespace["__file__"] = source_file
    stdout_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            prepared_code = preparse_file(code)
            exec(compile(prepared_code, source_file or "<sage_exec>", "exec"), namespace, namespace)
    except Exception as exc:
        emit(
            {
                "status": "error",
                "result_plain": "",
                "result_latex": "",
                "stdout": stdout_buffer.getvalue(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    result_obj = namespace.get(result_var)
    if result_obj is None and "result" in namespace:
        result_obj = namespace.get("result")

    emit(
        {
            "status": "ok",
            "result_plain": "" if result_obj is None else str(result_obj),
            "result_latex": "" if result_obj is None else str(latex(result_obj)),
            "result_data": to_result_data(result_obj),
            "stdout": stdout_buffer.getvalue(),
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit(
            {
                "status": "error",
                "result_plain": "",
                "result_latex": "",
                "stdout": "",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
"""


class SageRuntime:
    """Executes Sage code inside a constrained Docker container.

    Args:
        config: Resolved Docker and execution limits for Sage runs.
    """

    def __init__(self, config: SageRuntimeConfig, logger: ConsoleLogger | None = None):
        if not config.image:
            raise ValueError("Sage runtime requires a Docker image.")
        self.config = config
        self.logger = logger or ConsoleLogger()

    def _progress(self, message: str) -> None:
        if not self.config.progress_logs:
            return
        self.logger.progress(message)

    def execute_sage_code(
        self,
        code: str,
        timeout_sec: float | None = None,
        result_var: str = "RESULT",
    ) -> ExecutionResult:
        return self._execute(code=code, result_var=result_var, timeout_sec=timeout_sec)

    def _execute(self, code: str, result_var: str, timeout_sec: float | None = None) -> ExecutionResult:
        """Executes the given Sage code in a Docker container and returns the result along with execution details."""
        try:
            source_file = self._write_source_file(code)
        except OSError as exc:
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                result_data=None,
                runtime_ms=0,
                stdout="",
                stderr="",
                error=f"Failed to write Sage source file: {exc}",
                error_kind="docker_error",
            )

        payload_json = json.dumps({"source_file": CONTAINER_SOURCE_FILE, "result_var": result_var}, ensure_ascii=True)
        container_name = f"llmxcas-sage-{uuid.uuid4().hex[:12]}"
        cmd = self._build_docker_cmd(payload_json, source_file, container_name=container_name)
        timeout = timeout_sec if timeout_sec is not None else self.config.wall_timeout_sec

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            runtime_ms = int((time.perf_counter() - started) * 1000)
            self._kill_container(container_name)
            return ExecutionResult(
                status="timeout",
                result_plain="",
                result_latex="",
                result_data=None,
                runtime_ms=runtime_ms,
                stdout="",
                stderr="",
                error="Execution timed out.",
                error_kind="timeout",
            )
        except OSError as exc:
            runtime_ms = int((time.perf_counter() - started) * 1000)
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                result_data=None,
                runtime_ms=runtime_ms,
                stdout="",
                stderr="",
                error=f"Failed to invoke docker: {exc}",
                error_kind="docker_error",
            )
        finally:
            try:
                source_file.unlink()
            except FileNotFoundError:
                pass

        runtime_ms = int((time.perf_counter() - started) * 1000)
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""

        # Guard against excessively large outputs that could cause issues for the host process.
        if len(stdout.encode("utf-8")) > self.config.output_max_bytes:
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                result_data=None,
                runtime_ms=runtime_ms,
                stdout="",
                stderr=stderr,
                error="Output exceeds output_max_bytes.",
                error_kind="invalid_runner_output",
                exit_code=completed.returncode,
            )

        if self.config.progress_logs:
            self._progress(f"Execution completed in {runtime_ms} ms with stdout: {stdout!r}...")
            self._progress(f"Execution exit code: {completed.returncode}")
            self._progress(f"Execution stderr: {stderr!r}")

        parsed = self._parse_runner_output(stdout)

        if parsed is None:
            if stderr.strip():
                error_msg = stderr.strip()
            elif completed.returncode != 0:
                error_msg = f"Sage process exited with code {completed.returncode} before producing JSON output."
            else:
                error_msg = "Runner output was not valid JSON."
            return ExecutionResult(
                status="error",
                result_plain="",
                result_latex="",
                result_data=None,
                runtime_ms=runtime_ms,
                stdout=stdout,
                stderr=stderr,
                error=error_msg,
                error_kind=self._classify_missing_json_failure(stderr=stderr, exit_code=completed.returncode),
                exit_code=completed.returncode,
            )

        status = str(parsed.get("status", "error"))
        error_value = parsed.get("error", "")
        error = str(error_value) if error_value else ""

        if completed.returncode != 0 and status == "ok":
            status = "error"
            if not error:
                error = stderr.strip() or f"Exit code {completed.returncode}"

        error_kind = self._classify_parsed_failure(
            status=status,
            error=error,
            stderr=stderr,
            exit_code=completed.returncode,
        )

        return ExecutionResult(
            status=status,
            result_plain=str(parsed.get("result_plain", "")),
            result_latex=str(parsed.get("result_latex", "")),
            result_data=parsed.get("result_data"),
            runtime_ms=runtime_ms,
            stdout=str(parsed.get("stdout", "")),
            stderr=(f"{stderr}\n{parsed.get('traceback', '')}".strip() if parsed.get("traceback") else stderr),
            error=error,
            error_kind=error_kind,
            exit_code=completed.returncode,
        )

    def _build_docker_cmd(self, payload_json: str, source_file: Path, container_name: str | None = None) -> list[str]:
        """Builds the Docker command to execute the Sage code with the specified constraints and environment.

        Parameters:
            payload_json: The JSON string containing execution details to pass to the runner.
            source_file: The path to the temporary file containing the Sage code to execute.
        """
        cmd = ["docker", "run", "--rm"]
        if container_name:
            cmd.extend(["--name", container_name])
        cmd.extend(["--label", "llmxcas.sage_runtime=true"])
        cmd.extend(
            [
                "--mount",
                f"type=bind,src={source_file},dst={CONTAINER_SOURCE_FILE},readonly",
            ]
        )

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
        """Launch the wrapper instead of the generated Sage file directly.

        A direct ``sage -python generated_code.py`` call would execute the code,
        but it would not give the host one stable JSON record with captured
        stdout, traceback details, and the extracted ``RESULT`` value. The
        wrapper runs the exact saved ``.py`` file and emits the marker line that
        the host parser consumes.
        """
        entry = self.config.entrypoint.rsplit("/", maxsplit=1)[-1].lower()
        if entry in {"bash", "sh"}:
            shell_flag = "-lc" if entry == "bash" else "-c"
            shell_script = (
                'SAGE_BIN="sage"; '
                'if ! command -v "$SAGE_BIN" >/dev/null 2>&1; then '
                "for p in /home/sage/sage/sage /opt/sagemath/sage /usr/local/bin/sage /usr/bin/sage; do "
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
    def _kill_container(container_name: str) -> None:
        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except Exception:
            pass

    @staticmethod
    def _write_source_file(code: str) -> Path:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as handle:
            handle.write(code)
            return Path(handle.name)

    @staticmethod
    def _parse_runner_output(stdout: str) -> dict[str, Any] | None:
        text = stdout.strip()
        if not text:
            return None

        marker = "LLM_SAGE_RESULT:"

        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line:
                continue

            if marker in line:
                candidate = line.split(marker, 1)[1].strip()
            else:
                candidate = line

            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(payload, dict):
                return payload

        return None

    @staticmethod
    def _classify_missing_json_failure(*, stderr: str, exit_code: int) -> str:
        lowered = stderr.lower()
        if "cannot connect to the docker daemon" in lowered or "pull access denied" in lowered:
            return "docker_error"
        if "illegal instruction" in lowered or exit_code in {132, 137, 139}:
            return "runtime_crash"
        if exit_code != 0:
            return "runtime_crash"
        return "invalid_runner_output"

    @staticmethod
    def _classify_parsed_failure(*, status: str, error: str, stderr: str, exit_code: int) -> str:
        if status == "timeout":
            return "timeout"
        if status != "error":
            return ""

        lowered = f"{error}\n{stderr}".lower()
        if "cannot connect to the docker daemon" in lowered or "pull access denied" in lowered:
            return "docker_error"
        if "illegal instruction" in lowered or exit_code in {132, 137, 139}:
            return "runtime_crash"
        if exit_code != 0 and not lowered.strip():
            return "runtime_crash"
        return "code_error"
