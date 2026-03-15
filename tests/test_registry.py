import unittest

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.tools.registry import ToolRegistry  # noqa: E402
from src.tools.types import ToolDefinition, ToolResult, ToolSpec  # noqa: E402


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
