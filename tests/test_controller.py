from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.controller import AgentController, ControllerConfig
from src.tools.registry import ToolRegistry
from src.tools.types import ToolDefinition, ToolResult, ToolSpec


class _FakeCompletions:
    def __init__(self, outputs: list[str]):
        self._outputs = list(outputs)

    def create(self, **_: object):
        if not self._outputs:
            raise RuntimeError("No more fake outputs available")
        content = self._outputs.pop(0)
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self, outputs: list[str]):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(outputs))


class AgentControllerTests(unittest.TestCase):
    def test_tool_loop_then_finalize(self) -> None:
        client = _FakeClient(
            [
                '{"answer": "Candidate", "tool_call": {"name": "echo", "arguments": {"value": "4"}}}',
                '{"answer": "4", "tool_call": null}',
            ]
        )

        registry = ToolRegistry()
        calls: list[dict[str, object]] = []

        def _echo(args: dict[str, object]) -> ToolResult:
            calls.append(args)
            return ToolResult(ok=True, content=str(args.get("value", "")))

        registry.register(
            ToolDefinition(
                spec=ToolSpec(name="echo", description="Echo", input_schema={"type": "object"}),
                handler=_echo,
            )
        )

        controller = AgentController(
            client=client,
            model_name="fake",
            tool_registry=registry,
            config=ControllerConfig(max_turns=3, temperature=0.0),
        )
        result = controller.solve("What is 2+2?")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result.tool_traces), 1)

    def test_invalid_model_output_returns_raw_text(self) -> None:
        client = _FakeClient(["The answer is approximately 10."])
        controller = AgentController(
            client=client,
            model_name="fake",
            tool_registry=ToolRegistry(),
            config=ControllerConfig(max_turns=1, temperature=0.0),
        )
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "The answer is approximately 10.")
        self.assertEqual(result.stop_reason, "invalid_model_output")

    def test_multi_json_output_parses_first_object(self) -> None:
        content = (
            '{"answer": "ok", "tool_call": null}\n'
            '{"answer": "ignored", "tool_call": null}'
        )
        client = _FakeClient([content])
        controller = AgentController(client=client, model_name="fake", tool_registry=ToolRegistry())
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "ok")
        self.assertEqual(result.stop_reason, "finalized")


if __name__ == "__main__":
    unittest.main()
