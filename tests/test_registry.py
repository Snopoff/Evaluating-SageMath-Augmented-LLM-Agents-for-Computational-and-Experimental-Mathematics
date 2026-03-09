from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.registry import ToolRegistry
from src.tools.types import ToolDefinition, ToolResult, ToolSpec


class ToolRegistryTests(unittest.TestCase):
    def test_register_and_execute(self) -> None:
        registry = ToolRegistry()

        def _echo(args: dict[str, object]) -> ToolResult:
            return ToolResult(ok=True, content=str(args.get("value", "")))

        registry.register(
            ToolDefinition(
                spec=ToolSpec(
                    name="echo",
                    description="Echo tool",
                    input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                ),
                handler=_echo,
            )
        )

        result = registry.execute("echo", {"value": "hello"})
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "hello")
        self.assertEqual(len(registry.list_tools()), 1)
        self.assertEqual(registry.list_tools()[0].name, "echo")

    def test_unknown_tool(self) -> None:
        registry = ToolRegistry()
        result = registry.execute("missing", {})
        self.assertFalse(result.ok)
        self.assertIn("Unknown tool", result.content)


if __name__ == "__main__":
    unittest.main()
