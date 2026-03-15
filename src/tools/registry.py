from __future__ import annotations

from typing import Any

from src.tools.types import ToolDefinition, ToolResult, ToolSpec


class ToolRegistry:
    """Minimal registry for tool registration, lookup, and dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        name = tool.spec.name
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool name must be a non-empty string.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        if not callable(tool.handler):
            raise ValueError("Tool handler must be callable.")

        self._tools[name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(ok=False, content=f"Unknown tool: {name}")

        try:
            return tool.handler(dict(arguments))
        except Exception as exc:
            return ToolResult(ok=False, content=f"Tool error: {exc}")
