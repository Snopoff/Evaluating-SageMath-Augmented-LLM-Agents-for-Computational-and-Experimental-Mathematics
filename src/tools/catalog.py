from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from src.agent.schemas import (
    LeanExecArgs,
    SageExecArgs,
    SageFinalAnswerArgs,
)
from src.agent.verification import normalize_verification_payload
from src.lean.runtime import LeanRuntime
from src.sage.runtime import SageRuntime

SAGE_EXEC_TOOL_NAME = "sage_exec"
LEAN_EXEC_TOOL_NAME = "lean_exec"
FINAL_ANSWER_TOOL_NAME = "submit_final_answer"

EXEC_TOOL_NAMES = frozenset({SAGE_EXEC_TOOL_NAME, LEAN_EXEC_TOOL_NAME})


def make_sage_exec_tool(runtime: SageRuntime, usage_notes: str = "") -> BaseTool:
    description = (
        "Execute Sage script code inside Docker. Sage preparser syntax is allowed, "
        "including R.<x> declarations and ^ exponentiation. Assign the final value to RESULT."
    )
    if usage_notes.strip():
        description = f"{description}\n\nUsage notes:\n{usage_notes.strip()}"

    @tool(
        SAGE_EXEC_TOOL_NAME,
        description=description,
        args_schema=SageExecArgs,
        response_format="content_and_artifact",
    )
    def _sage_exec(code: str, result_var: str = "RESULT") -> tuple[str, dict[str, Any]]:
        result = runtime.execute_sage_code(
            code=code,
            result_var=result_var,
        )

        content = result.result_plain
        if not content and result.stdout.strip():
            content = result.stdout.strip()
        if result.status != "ok":
            content = result.error or result.stderr.strip() or "Sage execution failed"

        verification = normalize_verification_payload(result.result_data)
        artifact = {
            "ok": result.status == "ok",
            "status": result.status,
            "error_kind": getattr(result, "error_kind", ""),
            "exit_code": getattr(result, "exit_code", None),
            "runtime_ms": result.runtime_ms,
            "stderr": result.stderr,
            "result_latex": result.result_latex,
            "result_data": result.result_data,
            "verification": verification,
            "code": code,
            "result_var": result_var,
        }
        return content, artifact

    return _sage_exec


def make_lean_exec_tool(runtime: LeanRuntime, usage_notes: str = "") -> BaseTool:
    description = (
        "Elaborate Lean 4 code against Mathlib. Mathlib is already imported, so do not "
        "write `import` lines. Each call is stateless: every declaration you use must be "
        "defined in the same snippet. Pass `decl_name` naming the theorem you intend to "
        "prove -- its axioms are checked, and without it nothing counts as proved."
    )
    if usage_notes.strip():
        description = f"{description}\n\nUsage notes:\n{usage_notes.strip()}"

    @tool(
        LEAN_EXEC_TOOL_NAME,
        description=description,
        args_schema=LeanExecArgs,
        response_format="content_and_artifact",
    )
    def _lean_exec(code: str, decl_name: str = "") -> tuple[str, dict[str, Any]]:
        result = runtime.execute_lean_code(code=code, decl_name=decl_name)

        # What the model sees. `proved` is stated explicitly because a snippet can
        # elaborate cleanly (status "ok") while proving nothing -- `sorry` is only
        # a warning in Lean.
        lines = [f"status: {result.status}", f"proved: {result.proved}"]
        if result.error_kind:
            lines.append(f"error_kind: {result.error_kind}")
        if result.refuted:
            lines.append("refuted: Lean showed the proposition is false")
        if result.sorries:
            lines.append(f"unresolved sorries: {len(result.sorries)}")
        if decl_name:
            lines.append(f"axioms of {decl_name}: {result.axioms or 'none'}")
        diagnostics = result.stdout.strip() or result.error.strip()
        if diagnostics:
            lines.append("messages:\n" + diagnostics)
        content = "\n".join(lines)

        artifact = {
            "ok": result.status == "ok",
            "status": result.status,
            "proved": result.proved,
            "refuted": result.refuted,
            "error_kind": result.error_kind,
            "axioms": result.axioms,
            "has_sorry_axiom": result.has_sorry_axiom,
            "has_native_decide": result.has_native_decide,
            "sorry_count": len(result.sorries),
            "runtime_ms": result.runtime_ms,
            "stderr": result.stderr,
            "code": code,
            "decl_name": decl_name,
        }
        return content, artifact

    return _lean_exec


def make_submit_verdict_tool(args_schema: type[BaseModel]) -> BaseTool:
    """Final-answer tool for an arbitrary schema."""

    @tool(FINAL_ANSWER_TOOL_NAME, args_schema=args_schema)
    def _submit_verdict(**kwargs: Any) -> str:
        """Submit the structured final verdict."""
        return args_schema(**kwargs).model_dump_json()

    return _submit_verdict


def make_submit_final_answer_tool(args_schema: type[BaseModel] = SageFinalAnswerArgs) -> BaseTool:
    @tool(FINAL_ANSWER_TOOL_NAME, args_schema=args_schema)
    def _submit_final_answer(
        final_answer: str,
        sympy_answer: str | list[str],
        explanation: str,
        confidence: int,
        verified_claims: list[str] | None = None,
    ) -> str:
        """Submit the structured final answer to the math problem."""

        payload_kwargs: dict[str, Any] = {
            "final_answer": final_answer,
            "sympy_answer": sympy_answer,
            "explanation": explanation,
            "confidence": confidence,
        }
        if "verified_claims" in getattr(args_schema, "model_fields", {}):
            payload_kwargs["verified_claims"] = verified_claims

        payload = args_schema(**payload_kwargs)
        return payload.model_dump_json()

    return _submit_final_answer


AVAILABLE_TOOLS: dict[str, Callable[..., BaseTool]] = {
    SAGE_EXEC_TOOL_NAME: make_sage_exec_tool,
    LEAN_EXEC_TOOL_NAME: make_lean_exec_tool,
}
