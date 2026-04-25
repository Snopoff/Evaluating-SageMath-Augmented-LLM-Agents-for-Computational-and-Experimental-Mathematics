from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import BaseTool, tool

from src.agent.schemas import FinalAnswerArgs, SageExecArgs
from src.agent.verification import normalize_verification_payload
from src.sage.runtime import SageRuntime


SAGE_EXEC_TOOL_NAME = "sage_exec"
FINAL_ANSWER_TOOL_NAME = "submit_final_answer"


def make_sage_exec_tool(runtime: SageRuntime, usage_notes: str = "") -> BaseTool:
    description = "Execute raw Sage code inside Docker. Assign the final value to RESULT."
    if usage_notes.strip():
        description = f"{description}\n\nUsage notes:\n{usage_notes.strip()}"

    @tool(
        SAGE_EXEC_TOOL_NAME,
        description=description,
        args_schema=SageExecArgs,
        response_format="content_and_artifact",
    )
    def _sage_exec(code: str, result_var: str = "RESULT", timeout_sec: float | None = None) -> tuple[str, dict[str, Any]]:
        result = runtime.execute_sage_code(
            code=code,
            result_var=result_var,
            timeout_sec=timeout_sec,
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
    def _submit_final_answer(final_answer: str) -> str:
        """Submit the final answer to the math problem."""

        return final_answer

    return _submit_final_answer


AVAILABLE_TOOLS: dict[str, Callable[[SageRuntime, str], BaseTool]] = {
    SAGE_EXEC_TOOL_NAME: make_sage_exec_tool,
}
