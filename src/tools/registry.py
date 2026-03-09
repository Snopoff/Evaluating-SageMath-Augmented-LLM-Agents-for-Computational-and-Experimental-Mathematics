from __future__ import annotations

from typing import Any

from src.tools.types import ToolHandler, ToolResult, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, name: str, schema: dict[str, Any], handler: ToolHandler, description: str = "") -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        if not callable(handler):
            raise ValueError("Tool handler must be callable.")

        self._tools[name] = (
            ToolSpec(name=name, description=description or name, input_schema=dict(schema)),
            handler,
        )

    def get(self, name: str) -> tuple[ToolSpec, ToolHandler] | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return [spec for spec, _ in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        entry = self.get(name)
        if entry is None:
            return ToolResult(ok=False, content=f"Unknown tool: {name}")

        _, handler = entry
        try:
            return handler(dict(arguments))
        except Exception as exc:
            return ToolResult(ok=False, content=f"Tool error: {exc}")
