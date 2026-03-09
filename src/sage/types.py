from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SageRuntimeConfig:
    image: str
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
    status: str
    result_plain: str
    result_latex: str
    runtime_ms: int
    stdout: str
    stderr: str
    error: str = ""
    exit_code: int | None = None
