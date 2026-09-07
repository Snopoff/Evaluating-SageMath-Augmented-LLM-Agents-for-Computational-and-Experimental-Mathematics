"""Smoke tests for the Lean runtime.

These exercise a real Lean + Mathlib process, so the module-scoped runtime pays
`import Mathlib` (~35 s) exactly once for the whole file.

The most important test here is `test_sorry_is_not_a_proof`: Lean reports `sorry`
as a *warning*, so a snippet using it elaborates cleanly. Conflating
`status == "ok"` with a real proof is the one bug that would silently invalidate
every downstream verdict.
"""

from __future__ import annotations

import pytest

from src.lean.runtime import (
    LeanRuntime,
    parse_axioms,
    scan_for_import,
    scan_for_native_decide,
)
from src.lean.types import LeanRuntimeConfig
from src.benchmark.lean_pipeline import verify_certificate


@pytest.mark.parametrize("verdict,code,expected", [
    ("PROVED", "def candidate_claim : Prop := ∀ n : Nat, n + 0 = n\n"
     "theorem candidate_certificate : candidate_claim := by intro n; rfl", True),
    ("REFUTED", "def candidate_claim : Prop := 1 + 1 = (3 : Nat)\n"
     "theorem candidate_certificate : ¬ candidate_claim := by norm_num [candidate_claim]", True),
    ("PROVED", "def candidate_claim : Prop := 1 + 1 = (3 : Nat)\n"
     "theorem candidate_certificate : 1 + 1 = (2 : Nat) := by norm_num", False),
    ("PROVED", "def candidate_claim : Prop := 1 + 1 = (3 : Nat)\n"
     "theorem candidate_certificate : candidate_claim := by sorry", False),
    ("PROVED", "def candidate_claim : Prop := 1 + 1 = (3 : Nat)\n"
     "axiom invented : candidate_claim\n"
     "theorem candidate_certificate : candidate_claim := invented", False),
])
def test_pipeline_final_certificate(runtime, verdict, code, expected):
    result = verify_certificate(runtime, {"verdict": verdict, "lean_proof": code})
    assert result["proved"] is expected, result


@pytest.fixture(scope="module")
def runtime():
    lean_runtime = LeanRuntime(LeanRuntimeConfig(pool_size=1, wall_timeout_sec=60.0))
    # Warm the worker here so `import Mathlib` is not billed to whichever test
    # runs first -- otherwise per-test timings are meaningless.
    lean_runtime.execute_lean_code("theorem warmup_decl : 1 = 1 := rfl", "warmup_decl")
    try:
        yield lean_runtime
    finally:
        lean_runtime.shutdown()


def test_trivially_true_is_proved(runtime):
    result = runtime.execute_lean_code("theorem t1 : 1 + 1 = 2 := by decide", "t1")
    assert result.status == "ok"
    assert result.proved is True
    assert result.axioms == []
    assert result.error_kind == ""


def test_trivially_false_is_rejected_and_refuted(runtime):
    result = runtime.execute_lean_code("theorem t2 : 1 + 1 = 3 := by decide", "t2")
    assert result.status == "error"
    assert result.proved is False
    assert result.error_kind == "elab_error"
    # Lean explicitly announces the proposition is false -- a genuine refutation.
    assert result.refuted is True


def test_sorry_is_not_a_proof(runtime):
    """`sorry` elaborates cleanly, so status is ok -- but nothing was proved."""
    result = runtime.execute_lean_code(
        "theorem t3 : ∀ n : Nat, n + 0 = n := by sorry", "t3"
    )
    assert result.status == "ok"
    assert result.proved is False
    assert result.sorries
    assert result.has_sorry_axiom is True
    assert result.error_kind == "sorry_remaining"


def test_unknown_identifier_is_classified(runtime):
    result = runtime.execute_lean_code(
        "theorem t4 : True := by exact fooBarUnknownIdentifier", "t4"
    )
    assert result.proved is False
    assert result.error_kind == "unknown_identifier"


def test_timeout_kills_and_recovers(runtime):
    """A kernel-level `decide` blowup must time out, and the pool must survive it.

    `maxRecDepth`/`maxHeartbeats` are lifted deliberately: with them at their
    defaults Lean returns a fast error instead of hanging, which would not
    exercise the wall-clock path at all.
    """
    result = runtime.execute_lean_code(
        "set_option maxRecDepth 1000000 in\n"
        "set_option maxHeartbeats 0 in\n"
        "theorem t5 : Nat.Prime 1000003 := by decide",
        "t5",
        timeout_sec=8.0,
    )
    assert result.status == "timeout"
    assert result.error_kind == "timeout"
    assert result.proved is False
    # The worker was killed; the next call must transparently respawn it.
    recovered = runtime.execute_lean_code("theorem t5b : 2 + 2 = 4 := by decide", "t5b")
    assert recovered.proved is True


def test_native_decide_is_banned_without_touching_lean(runtime):
    """The static ban must fire before any Lean process is checked out.

    Reuses the module runtime on purpose: constructing a second one here would
    spawn another ~6 GB Lean worker for a test that must never reach Lean.
    """
    result = runtime.execute_lean_code(
        "theorem t6 : Nat.factorial 10 = 3628800 := by native_decide", "t6"
    )
    assert result.error_kind == "native_decide_banned"
    assert result.proved is False
    assert result.runtime_ms < 1000


def test_missing_decl_name_can_never_be_proved(runtime):
    """No declaration named means no axiom check, so no proof claim is allowed."""
    result = runtime.execute_lean_code("theorem t7 : 1 + 1 = 2 := by decide", "")
    assert result.status == "ok"
    assert result.proved is False


def test_import_lines_are_rejected_before_touching_lean(runtime):
    """Mathlib is pre-imported; an `import` line only produces an opaque error.

    Agents write them anyway, so the runtime intercepts and says what to do
    instead rather than letting the model debug a misleading message.
    """
    result = runtime.execute_lean_code(
        "import Mathlib.Data.Nat.Basic\ntheorem t8 : 1 + 1 = 2 := by decide", "t8"
    )
    assert result.error_kind == "import_not_allowed"
    assert result.proved is False
    assert result.runtime_ms < 1000
    assert "already imported" in result.error


def test_scan_for_import_ignores_the_word_elsewhere():
    assert scan_for_import("import Mathlib") is True
    assert scan_for_import("-- important\ndef important : Nat := 1") is False


def test_parse_axioms_shapes():
    assert parse_axioms("'t1' does not depend on any axioms") == []
    assert parse_axioms("'t3' depends on axioms: [sorryAx]") == ["sorryAx"]
    assert parse_axioms("'t' depends on axioms: [propext, Classical.choice]") == [
        "propext",
        "Classical.choice",
    ]


def test_scan_for_native_decide():
    assert scan_for_native_decide("by native_decide") is True
    assert scan_for_native_decide("by decide") is False
