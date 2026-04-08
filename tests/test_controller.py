import types
import unittest

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.agent.controller import AgentController, ControllerConfig  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402
from src.tools.types import ToolDefinition, ToolResult, ToolSpec  # noqa: E402
from src.utils.console_logging import ConsoleLogger  # noqa: E402


class _FakeCompletions:
    def __init__(self, outputs: list[object]):
        self._outputs = list(outputs)

    def create(self, **_: object):
        if not self._outputs:
            raise RuntimeError("No more fake outputs available")
        response_payload = self._outputs.pop(0)
        usage = None
        if isinstance(response_payload, tuple):
            content, usage_payload = response_payload
            usage = types.SimpleNamespace(**usage_payload)
        else:
            content = response_payload
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message)
        return types.SimpleNamespace(choices=[choice], usage=usage)


class _FakeClient:
    def __init__(self, outputs: list[str]):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(outputs))


class _RecordingLogger(ConsoleLogger):
    def __init__(self) -> None:
        super().__init__(mode="test")
        self.progress_messages: list[str] = []

    def progress(self, message: str) -> None:
        self.progress_messages.append(message)


def _verification_payload(
    *,
    summary: str = "pass",
    checks: list[dict[str, object]] | None = None,
    outputs: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "verification": {
            "summary": summary,
            "checks": list(checks or [{"id": "constraint_1", "status": "pass", "evidence": "verified"}]),
            "outputs": dict(outputs or {}),
        }
    }


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

        logger = _RecordingLogger()
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
            logger=logger,
        )

        result = controller.solve("What is 2+2?")
        self.assertEqual(result.final_answer, "4")
        messages = logger.progress_messages
        self.assertTrue(any("model reply:" in message for message in messages))
        self.assertTrue(any("tool call: echo" in message for message in messages))
        self.assertTrue(any("tool result: echo" in message for message in messages))

    def test_logger_records_raw_messages_tool_payloads_and_verified_code(self) -> None:
        code = "RESULT = {'verification': {'summary': 'pass', 'checks': [{'id': 'constraint_1', 'status': 'pass', 'evidence': 'ok'}], 'outputs': {}}, 'value': 4}"
        first_response = f'{{"answer": "Verify it", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{code}"}}}}}}'
        client = _FakeClient(
            [
                (first_response, {"prompt_tokens": 31, "completion_tokens": 9, "total_tokens": 40}),
                ('{"answer": "4", "tool_call": null}', {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}),
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(_: dict[str, object]) -> ToolResult:
            return ToolResult(ok=True, content="4", metadata={"status": "ok", "verification": _verification_payload()["verification"], "result_data": _verification_payload()})

        registry.register(
            ToolDefinition(
                spec=ToolSpec(name="sage_exec", description="Execute Sage", input_schema={"type": "object"}),
                handler=_sage_exec,
            )
        )

        logger = _RecordingLogger()
        controller = AgentController(
            client=client,
            model_name="fake",
            tool_registry=registry,
            config=ControllerConfig(max_steps=2, require_verification_for_final=True),
            logger=logger,
        )

        result = controller.solve("Q")

        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(logger.run_metadata["question"], "Q")
        self.assertIn("system_prompt", logger.run_metadata)

        model_events = [event for event in logger.events if event["kind"] == "model_call"]
        self.assertEqual(model_events[0]["payload"]["messages"][1]["content"], "Q")
        self.assertEqual(model_events[0]["payload"]["raw_response"], first_response)
        self.assertEqual(model_events[0]["payload"]["parsed_payload"]["tool_call"]["name"], "sage_exec")
        self.assertEqual(model_events[0]["payload"]["token_usage"]["input_tokens"], 31)
        self.assertEqual(model_events[0]["payload"]["token_usage"]["output_tokens"], 9)
        self.assertEqual(model_events[0]["payload"]["token_usage"]["total_tokens"], 40)

        tool_call_events = [event for event in logger.events if event["kind"] == "tool_call"]
        self.assertEqual(tool_call_events[0]["payload"]["arguments"]["code"], code)

        tool_result_events = [event for event in logger.events if event["kind"] == "tool_result"]
        self.assertTrue(tool_result_events[0]["payload"]["ok"])
        self.assertEqual(tool_result_events[0]["payload"]["metadata"]["verification"]["summary"], "pass")

        self.assertEqual(logger.run_metadata["agent_id"], "single_agent")
        self.assertEqual(logger.run_metadata["agent_ids"], ["single_agent"])
        self.assertEqual(logger.token_usage_totals["input_tokens"], 43)
        self.assertEqual(logger.token_usage_totals["output_tokens"], 13)
        self.assertEqual(logger.token_usage_totals["total_tokens"], 56)
        self.assertEqual(logger.final_result["verified_sage_code"], code)
        self.assertEqual(logger.final_results["single_agent"]["verified_sage_code"], code)

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
            return ToolResult(
                ok=True,
                content="4",
                metadata={"status": "ok", "verification": _verification_payload()["verification"], "result_data": _verification_payload()},
            )

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
            summary = "pass" if code == second_code else "fail"
            payload = _verification_payload(summary=summary)
            return ToolResult(ok=True, content=content, metadata={"status": "ok", "verification": payload["verification"], "result_data": payload})

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
        code = "RESULT = {'verification': {'summary': 'fail', 'checks': [{'id': 'constraint_1', 'status': 'fail', 'evidence': 'bad'}], 'outputs': {}}}"
        client = _FakeClient(
            [
                f'{{"answer": "Tried verification", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{code}"}}}}}}',
                '{"answer": "Still done", "tool_call": null}',
                '{"answer": "Out of steps", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(_: dict[str, object]) -> ToolResult:
            payload = _verification_payload(summary="fail", checks=[{"id": "constraint_1", "status": "fail", "evidence": "bad"}])
            return ToolResult(ok=True, content="{}", metadata={"status": "ok", "verification": payload["verification"], "result_data": payload})

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
        code = "RESULT = {'verification': {'summary': 'pass', 'checks': [{'id': 'constraint_1', 'status': 'pass', 'evidence': 'ok'}], 'outputs': {}}, 'value': 4}"
        client = _FakeClient(
            [
                f'{{"answer": "Verified", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{code}"}}}}}}',
                '{"answer": "4", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(_: dict[str, object]) -> ToolResult:
            payload = _verification_payload()
            return ToolResult(ok=True, content="4", metadata={"status": "ok", "verification": payload["verification"], "result_data": payload})

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

    def test_prepass_requires_full_constraint_coverage(self) -> None:
        code = "RESULT = {'verification': {'summary': 'pass', 'checks': [{'id': 'constraint_1', 'status': 'pass', 'evidence': 'ok'}], 'outputs': {'output_1': 19}}}"
        client = _FakeClient(
            [
                '{"hard_constraints": [{"id": "constraint_1", "text": "Must be odd", "requires_cas": true}, {"id": "constraint_2", "text": "Must be monic", "requires_cas": true}], "required_outputs": [{"id": "output_1", "text": "Compute p(19)"}]}',
                f'{{"answer": "Verified", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{code}"}}}}}}',
                '{"answer": "Done", "tool_call": null}',
                '{"answer": "Out of steps", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(_: dict[str, object]) -> ToolResult:
            payload = _verification_payload(
                checks=[{"id": "constraint_1", "status": "pass", "evidence": "ok"}],
                outputs={"output_1": 19},
            )
            return ToolResult(ok=True, content="4", metadata={"status": "ok", "verification": payload["verification"], "result_data": payload})

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
                require_verification_for_final=True,
                extract_constraints_before_solve=True,
                require_full_constraint_coverage=True,
            ),
        )

        result = controller.solve("Q")

        self.assertEqual(result.stop_reason, "max_steps_reached")
        self.assertEqual(result.verified_sage_code, "")

    def test_rejects_finalization_when_answer_admits_failed_constraint(self) -> None:
        code = "RESULT = {'verification': {'summary': 'pass', 'checks': [{'id': 'constraint_1', 'status': 'pass', 'evidence': 'ok'}], 'outputs': {}}}"
        client = _FakeClient(
            [
                f'{{"answer": "Verified", "tool_call": {{"name": "sage_exec", "arguments": {{"code": "{code}"}}}}}}',
                '{"answer": "The constraint is not satisfied.", "tool_call": null}',
                '{"answer": "Out of steps", "tool_call": null}',
            ]
        )
        registry = ToolRegistry()

        def _sage_exec(_: dict[str, object]) -> ToolResult:
            payload = _verification_payload()
            return ToolResult(ok=True, content="4", metadata={"status": "ok", "verification": payload["verification"], "result_data": payload})

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

        self.assertEqual(result.stop_reason, "max_steps_reached")

    def test_system_prompt_includes_usage_notes_when_present(self) -> None:
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                spec=ToolSpec(
                    name="sage_exec",
                    description="Execute Sage",
                    input_schema={"type": "object"},
                    usage_notes="Use RESULT.",
                ),
                handler=lambda _: ToolResult(ok=True, content="ok"),
            )
        )

        controller = AgentController(client=_FakeClient(['{"answer": "ok", "tool_call": null}']), model_name="fake", tool_registry=registry)

        self.assertIn("usage_notes: Use RESULT.", controller._system_prompt())


if __name__ == "__main__":
    unittest.main()
