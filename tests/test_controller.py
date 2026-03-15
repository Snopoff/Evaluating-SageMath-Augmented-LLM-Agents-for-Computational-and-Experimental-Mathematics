import types
import unittest
from unittest.mock import patch

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.agent.controller import AgentController, ControllerConfig  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402
from src.tools.types import ToolDefinition, ToolResult, ToolSpec  # noqa: E402


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
            config=ControllerConfig(max_steps=3, temperature=0.0, require_successful_tool_call_for_final=False),
        )
        result = controller.solve("What is 2+2?")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result.tool_traces), 1)
        self.assertEqual(result.verified_sage_code, "")

    def test_invalid_model_output_returns_raw_text(self) -> None:
        client = _FakeClient(["The answer is approximately 10."])
        controller = AgentController(
            client=client,
            model_name="fake",
            tool_registry=ToolRegistry(),
            config=ControllerConfig(max_steps=1, temperature=0.0),
        )
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "The answer is approximately 10.")
        self.assertEqual(result.stop_reason, "invalid_model_output")

    def test_multi_json_output_parses_first_object(self) -> None:
        content = '{"answer": "ok", "tool_call": null}\n{"answer": "ignored", "tool_call": null}'
        client = _FakeClient([content])
        controller = AgentController(client=client, model_name="fake", tool_registry=ToolRegistry())
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "ok")
        self.assertEqual(result.stop_reason, "finalized")

    def test_progress_logs_include_model_reply_and_tool_call(self) -> None:
        client = _FakeClient(
            [
                '{"answer": "Candidate", "tool_call": {"name": "echo", "arguments": {"value": "4"}}}',
                '{"answer": "4", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _echo(args: dict[str, object]) -> ToolResult:
            return ToolResult(ok=True, content=str(args.get("value", "")), metadata={"status": "ok"})

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
            config=ControllerConfig(
                max_steps=3,
                temperature=0.0,
                progress_logs=True,
                require_successful_tool_call_for_final=False,
            ),
        )

        with patch("src.agent.controller.progress") as mocked_progress:
            result = controller.solve("What is 2+2?")

        self.assertEqual(result.final_answer, "4")
        messages = [call.args[0] for call in mocked_progress.call_args_list]
        self.assertTrue(any("model reply:" in message for message in messages))
        self.assertTrue(any("tool call: echo" in message for message in messages))
        self.assertTrue(any("tool result: echo" in message for message in messages))

    def test_rejects_finalization_until_successful_sage_exec(self) -> None:
        code = "RESULT = 2 + 2"
        client = _FakeClient(
            [
                '{"answer": "Need verification", "tool_call": null}',
                f'{{"answer": "Running Sage", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{code}"}}}}}}',
                '{"answer": "4", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(args: dict[str, object]) -> ToolResult:
            return ToolResult(ok=True, content="4", metadata={"status": "ok", "result_data": {"verified": True}})

        registry.register(
            ToolDefinition(
                spec=ToolSpec(name="sage_exec", description="Execute Sage", input_schema={"type": "object"}),
                handler=_sage_exec,
            )
        )

        controller = AgentController(
            client=client,
            model_name="fake",
            tool_registry=registry,
            config=ControllerConfig(
                max_steps=3,
                require_successful_tool_call_for_final=True,
                require_verification_for_final=True,
            ),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(len(result.tool_traces), 1)
        self.assertEqual(result.verified_sage_code, code)

    def test_final_result_uses_last_successful_sage_code(self) -> None:
        first_code = "RESULT = 2 + 2"
        second_code = "RESULT = 3 + 3"
        client = _FakeClient(
            [
                f'{{"answer": "First attempt", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{first_code}"}}}}}}',
                f'{{"answer": "Second attempt", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{second_code}"}}}}}}',
                '{"answer": "6", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(args: dict[str, object]) -> ToolResult:
            code = args.get("code", "")
            content = "6" if code == second_code else "4"
            verified = code == second_code
            return ToolResult(ok=True, content=content, metadata={"status": "ok", "result_data": {"verified": verified}})

        registry.register(
            ToolDefinition(
                spec=ToolSpec(name="sage_exec", description="Execute Sage", input_schema={"type": "object"}),
                handler=_sage_exec,
            )
        )

        controller = AgentController(
            client=client,
            model_name="fake",
            tool_registry=registry,
            config=ControllerConfig(
                max_steps=3,
                require_successful_tool_call_for_final=True,
                require_verification_for_final=True,
            ),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "6")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(result.verified_sage_code, second_code)

    def test_rejects_finalization_when_last_successful_sage_result_is_not_verified(self) -> None:
        code = "RESULT = {'verified': False}"
        client = _FakeClient(
            [
                f'{{"answer": "Tried verification", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{code}"}}}}}}',
                '{"answer": "Still done", "tool_call": null}',
                '{"answer": "Out of steps", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(_: dict[str, object]) -> ToolResult:
            return ToolResult(ok=True, content="{}", metadata={"status": "ok", "result_data": {"verified": False}})

        registry.register(
            ToolDefinition(
                spec=ToolSpec(name="sage_exec", description="Execute Sage", input_schema={"type": "object"}),
                handler=_sage_exec,
            )
        )

        controller = AgentController(
            client=client,
            model_name="fake",
            tool_registry=registry,
            config=ControllerConfig(max_steps=3, require_verification_for_final=True),
        )

        result = controller.solve("Q")

        self.assertEqual(result.stop_reason, "max_steps_reached")
        self.assertEqual(result.verified_sage_code, "")

    def test_rejects_finalization_when_last_successful_sage_result_has_no_structured_verification(self) -> None:
        code = "RESULT = 4"
        client = _FakeClient(
            [
                f'{{"answer": "Computed something", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{code}"}}}}}}',
                '{"answer": "Done now", "tool_call": null}',
                '{"answer": "Out of steps", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(_: dict[str, object]) -> ToolResult:
            return ToolResult(ok=True, content="4", metadata={"status": "ok", "result_data": None})

        registry.register(
            ToolDefinition(
                spec=ToolSpec(name="sage_exec", description="Execute Sage", input_schema={"type": "object"}),
                handler=_sage_exec,
            )
        )

        controller = AgentController(
            client=client,
            model_name="fake",
            tool_registry=registry,
            config=ControllerConfig(max_steps=3, require_verification_for_final=True),
        )

        result = controller.solve("Q")

        self.assertEqual(result.stop_reason, "max_steps_reached")
        self.assertEqual(result.verified_sage_code, "")

    def test_accepts_finalization_when_last_successful_sage_result_is_verified(self) -> None:
        code = "RESULT = {'verified': True, 'value': 4}"
        client = _FakeClient(
            [
                f'{{"answer": "Verified", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{code}"}}}}}}',
                '{"answer": "4", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(_: dict[str, object]) -> ToolResult:
            return ToolResult(ok=True, content="4", metadata={"status": "ok", "result_data": {"verified": True, "value": 4}})

        registry.register(
            ToolDefinition(
                spec=ToolSpec(name="sage_exec", description="Execute Sage", input_schema={"type": "object"}),
                handler=_sage_exec,
            )
        )

        controller = AgentController(
            client=client,
            model_name="fake",
            tool_registry=registry,
            config=ControllerConfig(max_steps=2, require_verification_for_final=True),
        )

        result = controller.solve("Q")

        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.verified_sage_code, code)


if __name__ == "__main__":
    unittest.main()
