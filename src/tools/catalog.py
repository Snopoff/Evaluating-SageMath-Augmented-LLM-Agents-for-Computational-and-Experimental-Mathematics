from typing import Any, Callable

from src.sage.runtime import SageRuntime
from src.tools.types import ToolDefinition, ToolResult, ToolSpec

_SAGE_EXEC_SCHEMA = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "result_var": {"type": "string"},
        "timeout_sec": {"type": "number"},
    },
    "required": ["code"],
}


def _run_sage_tool(
    runtime: SageRuntime,
    *,
    code: str,
    result_var: str = "RESULT",
    timeout_sec: float | None = None,
) -> ToolResult:
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

    return ToolResult(
        ok=result.status == "ok",
        content=content,
        metadata={
            "status": result.status,
            "runtime_ms": result.runtime_ms,
            "stderr": result.stderr,
            "result_latex": result.result_latex,
        },
    )


def make_sage_exec_tool(runtime: SageRuntime) -> ToolDefinition:
    def _sage_exec_handler(arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(ok=False, content="sage_exec requires 'code' as a non-empty string")

        result_var = arguments.get("result_var", "RESULT")
        if not isinstance(result_var, str) or not result_var.strip():
            result_var = "RESULT"

        timeout = arguments.get("timeout_sec")
        timeout_sec: float | None = float(timeout) if isinstance(timeout, (int, float)) else None

        return _run_sage_tool(
            runtime=runtime,
            code=code,
            result_var=result_var,
            timeout_sec=timeout_sec,
        )

    return ToolDefinition(
        spec=ToolSpec(
            name="sage_exec",
            description="Execute raw Sage code inside Docker.",
            input_schema=dict(_SAGE_EXEC_SCHEMA),
        ),
        handler=_sage_exec_handler,
    )


AVAILABLE_TOOLS: dict[str, Callable[[SageRuntime], ToolDefinition]] = {
    "sage_exec": make_sage_exec_tool,
}
