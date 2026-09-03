"""Types for the Lean 4 runtime."""

from __future__ import annotations

from dataclasses import dataclass, field

#: Axioms that Mathlib itself is built on. A proof depending only on these is sound.
DEFAULT_AXIOM_ALLOWLIST = ("propext", "Classical.choice", "Quot.sound")

#: Axiom introduced by `native_decide`; trusts the compiler rather than the kernel.
NATIVE_DECIDE_AXIOM = "Lean.ofReduceBool"


@dataclass(frozen=True)
class LeanRuntimeConfig:
    lean_ws_dir: str = "lean_ws"
    repl_binary_path: str = "tools/lean-repl/.lake/build/bin/repl"
    backend: str = "repl"  # "repl" | "oneshot"
    # One worker by default: `import Mathlib` costs ~6 GB of RSS and this runs on a
    # 24 GB laptop that the user needs at the same time. Raise only deliberately.
    pool_size: int = 1
    startup_timeout_sec: float = 180.0
    wall_timeout_sec: float = 60.0
    max_output_chars: int = 20_000
    max_restarts: int = 3
    memory_soft_limit_mb: int = 7_000
    memory_poll_interval_sec: float = 5.0
    ban_native_decide: bool = True
    axiom_allowlist: tuple[str, ...] = DEFAULT_AXIOM_ALLOWLIST
    progress_logs: bool = False


@dataclass(frozen=True)
class LeanExecutionResult:
    status: str  # "ok" | "error" | "timeout"
    proved: bool = False
    messages: list[dict] = field(default_factory=list)
    sorries: list[dict] = field(default_factory=list)
    axioms: list[str] = field(default_factory=list)
    has_sorry_axiom: bool = False
    has_native_decide: bool = False
    refuted: bool = False
    declarations_checked: list[str] = field(default_factory=list)
    runtime_ms: int = 0
    stdout: str = ""
    stderr: str = ""
    error: str = ""
    error_kind: str = ""
    exit_code: int | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"
