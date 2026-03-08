from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmxm2.tools.registry import ToolRegistry
from llmxm2.tools.types import ToolResult


class ToolRegistryTests(unittest.TestCase):
    def test_register_and_execute(self) -> None:
        registry = ToolRegistry()

        def _echo(args: dict[str, object]) -> ToolResult:
            return ToolResult(ok=True, content=str(args.get("value", "")))

        registry.register(
            name="echo",
            schema={"type": "object", "properties": {"value": {"type": "string"}}},
            handler=_echo,
            description="Echo tool",
        )

        result = registry.execute("echo", {"value": "hello"})
        self.assertTrue(result.ok)
        self.assertEqual(result.content, "hello")
        self.assertEqual(len(registry.list_tools()), 1)

    def test_unknown_tool(self) -> None:
        registry = ToolRegistry()
        result = registry.execute("missing", {})
        self.assertFalse(result.ok)
        self.assertIn("Unknown tool", result.content)


if __name__ == "__main__":
    unittest.main()
