from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool, tool

from src.agent.schemas import FinalAnswerArgs, SageExecArgs
from src.agent.verification import normalize_verification_payload
from src.sage.runtime import SageRuntime


SAGE_EXEC_TOOL_NAME = "sage_exec"
FINAL_ANSWER_TOOL_NAME = "submit_final_answer"


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


def make_submit_final_answer_tool() -> BaseTool:
    @tool(FINAL_ANSWER_TOOL_NAME, args_schema=FinalAnswerArgs)
    def _submit_final_answer(final_answer: str, explanation: str, verified_claims: list[str] | None = None) -> str:
        """Submit the structured final answer to the math problem."""

        payload = FinalAnswerArgs(
            final_answer=final_answer,
            explanation=explanation,
            verified_claims=verified_claims or [],
        )
        return payload.model_dump_json()

    return _submit_final_answer


AVAILABLE_TOOLS: dict[str, Callable[[SageRuntime, str], BaseTool]] = {
    SAGE_EXEC_TOOL_NAME: make_sage_exec_tool,
}
