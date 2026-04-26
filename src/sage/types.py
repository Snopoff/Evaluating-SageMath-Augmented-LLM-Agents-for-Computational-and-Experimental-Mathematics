from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SageRuntimeConfig:
    """Configuration for the isolated Docker-backed Sage runtime.

    Args:
        image: Docker image reference used for Sage execution.
        platform: Optional Docker platform override such as ``linux/amd64``.
        entrypoint: Container entrypoint used to launch the Sage process.
        cpus: CPU quota passed to Docker.
        memory: Memory limit passed to Docker.
        pids_limit: Maximum number of processes allowed in the container.
        wall_timeout_sec: Default wall-clock timeout for each execution.
        cpu_limit_sec: CPU time limit enforced via Docker ulimit.
        output_max_bytes: Maximum allowed payload or stdout size in bytes.
        user: Optional Docker user override.
        home_dir: HOME value injected into the container.
        dot_sage_dir: DOT_SAGE value injected into the container.
        progress_logs: Whether runtime-level progress messages are enabled.
    """

    image: str = "docker.io/sagemath/sagemath:latest"
    platform: str = ""
    entrypoint: str = "/bin/bash"
    cpus: float = 1.0
    memory: str = "1g"
    pids_limit: int = 128
    wall_timeout_sec: float = 8.0
    cpu_limit_sec: float = 5.0
    output_max_bytes: int = 262_144
    user: str = ""
    home_dir: str = "/tmp"
    dot_sage_dir: str = "/tmp/.sage"
    progress_logs: bool = False


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome returned by a single Sage execution attempt.

    Args:
        status: Execution status such as ``ok``, ``error``, or ``timeout``.
        result_plain: Plain-text rendering of the computed result.
        result_latex: LaTeX rendering of the computed result.
        result_data: JSON-friendly structured rendering of the computed result
            when available.
        runtime_ms: End-to-end execution time in milliseconds.
        stdout: Captured stdout emitted by the Sage runner.
        stderr: Captured stderr emitted by Docker or the runner.
        error: Human-readable error message when execution fails.
        error_kind: Normalized failure class such as ``code_error`` or
            ``runtime_crash``.
        exit_code: Optional process exit code from the Docker command.
    """

    status: str
    result_plain: str
    result_latex: str
    result_data: object | None
    runtime_ms: int
    stdout: str
    stderr: str
    error: str = ""
    error_kind: str = ""
    exit_code: int | None = None
