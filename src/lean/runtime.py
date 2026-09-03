"""Lean 4 + Mathlib runtime."""

from __future__ import annotations

import atexit
import json
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Protocol

from src.lean.types import (
    NATIVE_DECIDE_AXIOM,
    LeanExecutionResult,
    LeanRuntimeConfig,
)

#: Every live runtime, so a hard stop can still reclaim its Lean workers.
_LIVE_RUNTIMES: list[LeanRuntime] = []


def shutdown_all() -> None:
    """Tear down every Lean worker this interpreter started."""
    for runtime in _LIVE_RUNTIMES:
        runtime.shutdown()


# `#print axioms foo` prints one of these two shapes.
NO_AXIOMS_RE = re.compile(r"does not depend on any axioms")
AXIOMS_RE = re.compile(r"depends on axioms:\s*\[(.*?)\]", re.DOTALL)

# Lean says "Unknown identifier `x`" / "unknown constant" -- match case-insensitively.
UNKNOWN_ID_RE = re.compile(r"unknown (identifier|constant|tactic)", re.IGNORECASE)

# `decide`/`norm_num` announcing that the goal is actually FALSE. This is a
# genuine refutation signal, not merely a failure to prove.
REFUTATION_RE = re.compile(r"proved that the proposition.*?is false", re.DOTALL | re.IGNORECASE)

# The REPL only accepts `import` when no environment is given, and we always run
# inside the pre-imported Mathlib environment. An import line therefore fails with
# an opaque elaboration error; catch it early and say what to do instead.
IMPORT_LINE_RE = re.compile(r"^\s*import\s+\S", re.MULTILINE)

# Ways to reach the kernel-bypassing evaluator. Banned outright.
NATIVE_DECIDE_PATTERNS = ("native_decide", "ofReduceBool", "Lean.ofReduceBool")

# Compiler diagnostics from the one-shot backend: `file.lean:12:4: error: ...`
ONESHOT_MSG_RE = re.compile(r"^(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+):\s*(?P<severity>error|warning|info):\s*(?P<data>.*)$")


def scan_for_native_decide(code: str) -> bool:
    """Cheap static check, run before any Lean process is touched."""
    return any(pattern in code for pattern in NATIVE_DECIDE_PATTERNS)


def scan_for_import(code: str) -> bool:
    return bool(IMPORT_LINE_RE.search(code))


def parse_axioms(text: str) -> list[str]:
    """Parse the output of `#print axioms <name>` into a list of axiom names."""
    if NO_AXIOMS_RE.search(text):
        return []
    match = AXIOMS_RE.search(text)
    if not match:
        return []
    return [part.strip() for part in match.group(1).split(",") if part.strip()]


def format_messages(messages: list[dict], max_chars: int) -> tuple[str, bool]:
    """Render Lean diagnostics for the agent. Returns (text, was_truncated).

    The tail is kept rather than the head: the most specific error is usually last.
    """
    lines = []
    for message in messages:
        position = message.get("pos") or {}
        line = position.get("line", "?")
        column = position.get("column", "?")
        severity = message.get("severity", "info")
        data = str(message.get("data", "")).strip()
        lines.append(f"{line}:{column}: {severity}: {data}")
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text, False
    return "...[truncated]\n" + text[-max_chars:], True


def classify_error_kind(
    *,
    messages: list[dict],
    sorries: list[dict],
    has_sorry_axiom: bool,
    truncated: bool,
) -> str:
    """Map a completed Lean run onto a stable error taxonomy."""
    error_texts = [str(message.get("data", "")) for message in messages if message.get("severity") == "error"]
    if error_texts:
        joined = "\n".join(error_texts)
        if UNKNOWN_ID_RE.search(joined):
            return "unknown_identifier"
        return "elab_error"
    if sorries or has_sorry_axiom:
        return "sorry_remaining"
    if truncated:
        return "output_too_large"
    return ""


class LeanBackend(Protocol):
    def run(self, code: str, decl_name: str, timeout_sec: float) -> LeanExecutionResult: ...

    def restart(self) -> None: ...

    def is_healthy(self) -> bool: ...

    def shutdown(self) -> None: ...


class _ReplWorker:
    """One long-lived `repl` process with `import Mathlib` already elaborated."""

    def __init__(self, config: LeanRuntimeConfig, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.proc: subprocess.Popen | None = None
        self.base_env: int | None = None
        self._responses: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None
        self._stderr_buffer: list[str] = []
        self.needs_restart = False
        self.restart_count = 0

    # -- process lifecycle -------------------------------------------------

    def _spawn(self) -> None:
        binary = str(self.repo_root / self.config.repl_binary_path)
        workspace = str(self.repo_root / self.config.lean_ws_dir)
        # shell=False with an argv list: the Mathlib path contains spaces.
        # start_new_session puts `lake` and the `repl` it forks into their own
        # process group, so a timeout can kill the whole tree. Killing only the
        # direct child orphans the real Lean process, which keeps burning CPU and
        # ~6 GB of RSS and makes every later call crawl.
        self.proc = subprocess.Popen(
            ["lake", "env", binary],
            cwd=workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        # The queue is bound to THIS process and handed to its reader thread.
        # Reading it off `self` instead would let a dead process's reader push a
        # stale EOF into the queue of its replacement, poisoning the fresh worker.
        responses: queue.Queue = queue.Queue()
        self._responses = responses
        self._reader = threading.Thread(target=self._read_loop, args=(self.proc.stdout, responses), daemon=True)
        self._reader.start()
        threading.Thread(target=self._drain_stderr, args=(self.proc.stderr,), daemon=True).start()

    def _read_loop(self, stream, responses: queue.Queue) -> None:
        buffer = ""
        for line in stream:
            if line.strip() == "" and buffer.strip():
                try:
                    responses.put(json.loads(buffer))
                except json.JSONDecodeError:
                    responses.put({"__raw__": buffer})
                buffer = ""
            else:
                buffer += line
        responses.put({"__eof__": True})

    def _drain_stderr(self, stream) -> None:
        for line in stream:
            self._stderr_buffer.append(line)
            del self._stderr_buffer[:-50]

    def start(self) -> None:
        """Spawn and warm the worker. Raises nothing; sets needs_restart on failure."""
        try:
            self._spawn()
            response = self._request({"cmd": "import Mathlib"}, self.config.startup_timeout_sec)
            self.base_env = response.get("env")
            self.needs_restart = self.base_env is None
        except Exception:
            self.needs_restart = True

    def restart(self) -> None:
        self.kill()
        self.restart_count += 1
        self.needs_restart = False
        self.start()

    def kill(self) -> None:
        """Kill the whole process group, not just `lake`.

        `lake env repl` forks the actual Lean process; killing the parent alone
        leaves that grandchild running.
        """
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            pass
        self.proc = None
        self.base_env = None

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure_ready(self) -> None:
        if not self.is_alive() or self.needs_restart or self.base_env is None:
            self.restart()

    # -- protocol ----------------------------------------------------------

    def _request(self, payload: dict, timeout_sec: float) -> dict:
        """Send one JSON command and wait for its response. Raises on timeout/EOF."""
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload) + "\n\n")
        self.proc.stdin.flush()
        response = self._responses.get(timeout=timeout_sec)
        if response.get("__eof__"):
            raise RuntimeError("repl process exited")
        if "__raw__" in response:
            raise ValueError(f"unparseable repl output: {response['__raw__'][:400]}")
        return response

    def rss_bytes(self) -> int:
        """Total RSS of the process tree. `lake env` forks, so the parent alone lies."""
        if self.proc is None:
            return 0
        try:
            import psutil

            process = psutil.Process(self.proc.pid)
            total = process.memory_info().rss
            for child in process.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except Exception:
                    continue
            return total
        except Exception:
            return 0

    def stderr_tail(self) -> str:
        return "".join(self._stderr_buffer)[-4000:]


class _ReplBackend:
    """Pool of `_ReplWorker`s, checked out one request at a time."""

    def __init__(self, config: LeanRuntimeConfig, repo_root: Path, logger=None) -> None:
        self.config = config
        self.repo_root = repo_root
        self.logger = logger
        self._workers: queue.Queue = queue.Queue()
        self._all_workers: list[_ReplWorker] = []
        self._unhealthy = False
        self._lock = threading.Lock()
        for _ in range(max(1, config.pool_size)):
            worker = _ReplWorker(config, repo_root)
            self._all_workers.append(worker)
            self._workers.put(worker)

    def is_healthy(self) -> bool:
        return not self._unhealthy

    def restart(self) -> None:
        with self._lock:
            for worker in self._all_workers:
                worker.needs_restart = True
            self._unhealthy = False

    def shutdown(self) -> None:
        for worker in self._all_workers:
            worker.kill()

    def run(self, code: str, decl_name: str, timeout_sec: float) -> LeanExecutionResult:
        started = time.monotonic()
        if self._unhealthy:
            return LeanExecutionResult(
                status="error",
                error_kind="repl_crash",
                error="Lean runtime is unhealthy after repeated restarts.",
            )
        worker = self._workers.get()
        try:
            return self._run_on(worker, code, decl_name, timeout_sec, started)
        finally:
            self._workers.put(worker)

    def _run_on(
        self,
        worker: _ReplWorker,
        code: str,
        decl_name: str,
        timeout_sec: float,
        started: float,
    ) -> LeanExecutionResult:
        try:
            worker.ensure_ready()
        except Exception as exc:  # pragma: no cover - defensive
            return self._fail(worker, "repl_spawn_error", str(exc), started)
        if worker.base_env is None:
            if worker.restart_count > self.config.max_restarts:
                self._unhealthy = True
            return self._fail(worker, "startup_timeout", "import Mathlib did not complete", started)

        try:
            response = worker._request({"cmd": code, "env": worker.base_env}, timeout_sec)
        except queue.Empty:
            worker.kill()
            worker.needs_restart = True
            return self._fail(worker, "timeout", "Lean execution timed out.", started)
        except RuntimeError as exc:
            worker.needs_restart = True
            if worker.restart_count > self.config.max_restarts:
                self._unhealthy = True
            return self._fail(worker, "repl_crash", str(exc), started)
        except ValueError as exc:
            return self._fail(worker, "invalid_repl_output", str(exc), started)
        except Exception as exc:  # pragma: no cover - defensive
            worker.needs_restart = True
            return self._fail(worker, "repl_crash", str(exc), started)

        if worker.rss_bytes() > self.config.memory_soft_limit_mb * 1_000_000:
            worker.needs_restart = True

        return self._build_result(worker, response, decl_name, timeout_sec, started)

    def _build_result(
        self,
        worker: _ReplWorker,
        response: dict,
        decl_name: str,
        timeout_sec: float,
        started: float,
    ) -> LeanExecutionResult:
        messages = response.get("messages") or []
        sorries = response.get("sorries") or []
        derived_env = response.get("env")

        axioms: list[str] = []
        axiom_check_failed = False
        if decl_name and derived_env is not None:
            try:
                axiom_response = worker._request(
                    {"cmd": f"#print axioms {decl_name}", "env": derived_env},
                    timeout_sec,
                )
                axiom_text = "\n".join(str(m.get("data", "")) for m in (axiom_response.get("messages") or []))
                if not axiom_text.strip():
                    axiom_check_failed = True
                axioms = parse_axioms(axiom_text)
            except Exception:
                axiom_check_failed = True

        return _assemble(
            messages=messages,
            sorries=sorries,
            axioms=axioms,
            axiom_check_failed=axiom_check_failed,
            decl_name=decl_name,
            config=self.config,
            started=started,
            stderr=worker.stderr_tail(),
        )

    def _fail(self, worker: _ReplWorker, error_kind: str, error: str, started: float) -> LeanExecutionResult:
        status = "timeout" if error_kind == "timeout" else "error"
        return LeanExecutionResult(
            status=status,
            proved=False,
            error=error,
            error_kind=error_kind,
            stderr=worker.stderr_tail(),
            runtime_ms=int((time.monotonic() - started) * 1000),
        )


class _OneShotBackend:
    """Fallback: `lake env lean <tmpfile>` per call. ~35 s each, zero protocol risk."""

    def __init__(self, config: LeanRuntimeConfig, repo_root: Path, logger=None) -> None:
        self.config = config
        self.repo_root = repo_root
        self.logger = logger

    def is_healthy(self) -> bool:
        return True

    def restart(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def run(self, code: str, decl_name: str, timeout_sec: float) -> LeanExecutionResult:
        started = time.monotonic()
        workspace = self.repo_root / self.config.lean_ws_dir
        scratch = workspace / "_scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        source = "import Mathlib\n" + code + "\n"
        if decl_name:
            source += f"#print axioms {decl_name}\n"
        handle = tempfile.NamedTemporaryFile("w", suffix=".lean", dir=str(scratch), delete=False, encoding="utf-8")
        try:
            handle.write(source)
            handle.close()
            completed = subprocess.run(
                ["lake", "env", "lean", handle.name],
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return LeanExecutionResult(
                status="timeout",
                error_kind="timeout",
                error="Lean execution timed out.",
                runtime_ms=int((time.monotonic() - started) * 1000),
            )
        except OSError as exc:
            return LeanExecutionResult(
                status="error",
                error_kind="repl_spawn_error",
                error=str(exc),
                runtime_ms=int((time.monotonic() - started) * 1000),
            )
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass

        messages: list[dict] = []
        axiom_lines: list[str] = []
        for line in (completed.stdout + "\n" + completed.stderr).splitlines():
            match = ONESHOT_MSG_RE.match(line.strip())
            if match:
                messages.append(
                    {
                        "severity": match.group("severity"),
                        "data": match.group("data"),
                        "pos": {
                            "line": int(match.group("line")),
                            "column": int(match.group("col")),
                        },
                    }
                )
            elif "axiom" in line:
                axiom_lines.append(line)

        axiom_text = "\n".join(axiom_lines) + "\n" + completed.stdout
        axioms = parse_axioms(axiom_text) if decl_name else []
        sorries = [m for m in messages if "sorry" in str(m.get("data", "")).lower()]
        return _assemble(
            messages=messages,
            sorries=sorries,
            axioms=axioms,
            axiom_check_failed=bool(decl_name) and not axiom_text.strip(),
            decl_name=decl_name,
            config=self.config,
            started=started,
            stderr=completed.stderr[-4000:],
            stdout=completed.stdout[-4000:],
            exit_code=completed.returncode,
        )


def _assemble(
    *,
    messages: list[dict],
    sorries: list[dict],
    axioms: list[str],
    axiom_check_failed: bool,
    decl_name: str,
    config: LeanRuntimeConfig,
    started: float,
    stderr: str = "",
    stdout: str = "",
    exit_code: int | None = None,
) -> LeanExecutionResult:
    """Shared result construction, including the soundness gate."""
    text, truncated = format_messages(messages, config.max_output_chars)
    has_errors = any(m.get("severity") == "error" for m in messages)
    has_sorry_axiom = "sorryAx" in axioms
    has_native_decide = NATIVE_DECIDE_AXIOM in axioms or any("ofReduceBool" in axiom for axiom in axioms)
    error_text = "\n".join(str(m.get("data", "")) for m in messages if m.get("severity") == "error")
    refuted = bool(REFUTATION_RE.search(error_text))

    unexpected = [axiom for axiom in axioms if axiom not in config.axiom_allowlist and axiom != "sorryAx"]

    # The soundness gate. Every condition must hold.
    proved = (
        not has_errors
        and not sorries
        and bool(decl_name)
        and not has_sorry_axiom
        and not has_native_decide
        and not axiom_check_failed
        and not unexpected
    )

    error_kind = classify_error_kind(
        messages=messages,
        sorries=sorries,
        has_sorry_axiom=has_sorry_axiom,
        truncated=truncated,
    )
    if not error_kind and axiom_check_failed:
        error_kind = "axiom_check_failed"
    if not error_kind and unexpected:
        error_kind = "unexpected_axiom"
    if not error_kind and has_native_decide:
        error_kind = "native_decide_banned"

    return LeanExecutionResult(
        status="error" if has_errors else "ok",
        proved=proved,
        messages=messages,
        sorries=sorries,
        axioms=axioms,
        has_sorry_axiom=has_sorry_axiom,
        has_native_decide=has_native_decide,
        refuted=refuted,
        declarations_checked=[decl_name] if decl_name else [],
        runtime_ms=int((time.monotonic() - started) * 1000),
        stdout=stdout or text,
        stderr=stderr,
        error=error_text[: config.max_output_chars],
        error_kind=error_kind,
        exit_code=exit_code,
    )


class LeanRuntime:
    """Public entry point. Never raises."""

    def __init__(
        self,
        config: LeanRuntimeConfig,
        logger=None,
        repo_root: str | os.PathLike | None = None,
    ) -> None:
        self.config = config
        self.logger = logger
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        backend_cls = _ReplBackend if config.backend == "repl" else _OneShotBackend
        self._backend: LeanBackend = backend_cls(config, self.repo_root, logger)
        # Guaranteed teardown. Lean workers hold ~6 GB each; an interpreter that
        # exits without shutting them down leaves orphans that accumulate across
        # runs until the machine swaps itself to death.
        atexit.register(self.shutdown)
        _LIVE_RUNTIMES.append(self)

    def execute_lean_code(
        self,
        code: str,
        decl_name: str = "",
        timeout_sec: float | None = None,
    ) -> LeanExecutionResult:
        started = time.monotonic()
        if not code or not code.strip():
            return LeanExecutionResult(status="error", error_kind="elab_error", error="Empty Lean snippet.")
        if scan_for_import(code):
            return LeanExecutionResult(
                status="error",
                error_kind="import_not_allowed",
                error=(
                    "Remove the `import` lines. All of Mathlib is already imported in "
                    "this environment; `import` is only valid in a fresh one and will "
                    "always fail here. Just use the names directly."
                ),
                runtime_ms=int((time.monotonic() - started) * 1000),
            )
        if self.config.ban_native_decide and scan_for_native_decide(code):
            return LeanExecutionResult(
                status="error",
                proved=False,
                error_kind="native_decide_banned",
                error=(
                    "`native_decide` (and `ofReduceBool`) are banned: they bypass the "
                    "Lean kernel. Use `decide`, `norm_num`, or an explicit proof."
                ),
                runtime_ms=int((time.monotonic() - started) * 1000),
            )
        timeout = timeout_sec if timeout_sec is not None else self.config.wall_timeout_sec
        try:
            return self._backend.run(code, decl_name, timeout)
        except Exception as exc:  # pragma: no cover - the contract is "never raise"
            return LeanExecutionResult(
                status="error",
                error_kind="repl_crash",
                error=f"{type(exc).__name__}: {exc}",
                runtime_ms=int((time.monotonic() - started) * 1000),
            )

    def shutdown(self) -> None:
        try:
            self._backend.shutdown()
        except Exception:
            pass
        try:
            _LIVE_RUNTIMES.remove(self)
        except ValueError:
            pass
