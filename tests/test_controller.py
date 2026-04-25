import unittest
from typing import Any, Sequence

import rootutils
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, tool

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.agent.controller import AgentController, ControllerConfig  # noqa: E402
from src.tools.catalog import FINAL_ANSWER_TOOL_NAME, SAGE_EXEC_TOOL_NAME  # noqa: E402
from src.utils.console_logging import ConsoleLogger  # noqa: E402


class _FakeModel:
    def __init__(self, outputs: list[AIMessage]) -> None:
        self.outputs = list(outputs)
        self.bound_tools: list[BaseTool] = []
        self.bind_kwargs: dict[str, Any] = {}
        self.invocations: list[list[Any]] = []
        self.model_name = "fake-model"

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> "_FakeModel":
        self.bound_tools = list(tools)
        self.bind_kwargs = dict(kwargs)
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        if not self.outputs:
            raise RuntimeError("No more fake outputs available")
        return self.outputs.pop(0)


class _RecordingLogger(ConsoleLogger):
    def __init__(self) -> None:
        super().__init__(mode="test")
        self.progress_messages: list[str] = []

    def progress(self, message: str) -> None:
        self.progress_messages.append(message)


def _verification_payload(summary: str = "pass") -> dict[str, object]:
    return {
        "summary": summary,
        "checks": [{"id": "constraint_1", "status": summary, "evidence": "verified"}],
        "outputs": {},
    }


def _make_sage_tool(calls: list[dict[str, Any]] | None = None, *, verification_summary: str | None = None) -> BaseTool:
    @tool(SAGE_EXEC_TOOL_NAME, response_format="content_and_artifact")
    def sage_exec(code: str) -> tuple[str, dict[str, Any]]:
        """Execute Sage."""

        if calls is not None:
            calls.append({"code": code})
        verification = None if verification_summary is None else _verification_payload(verification_summary)
        return "4", {
            "ok": True,
            "status": "ok",
            "verification": verification,
            "code": code,
        }

    return sage_exec


class AgentControllerTests(unittest.TestCase):
    def test_tool_loop_then_submit_final_answer(self) -> None:
        code = "RESULT = 2 + 2"
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": code}, "id": "call_1"}]),
                AIMessage(content="", tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_2"}]),
            ]
        )
        calls: list[dict[str, Any]] = []
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool(calls)],
            config=ControllerConfig(max_steps=3),
        )

        result = controller.solve("What is 2+2?")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(calls, [{"code": code}])
        self.assertEqual(len(result.tool_traces), 1)
        self.assertEqual(result.verified_sage_code, code)
        self.assertFalse(model.bind_kwargs["parallel_tool_calls"])
        self.assertEqual({tool.name for tool in model.bound_tools}, {SAGE_EXEC_TOOL_NAME, FINAL_ANSWER_TOOL_NAME})

    def test_plain_text_is_reprompted_when_structured_final_required(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="4"),
                AIMessage(content="", tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_1"}]),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=2, require_structured_final=True),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(model.invocations[1][-1].content, f"Use the {FINAL_ANSWER_TOOL_NAME} tool to submit the final answer.")

    def test_plain_text_can_finalize_when_structured_final_not_required(self) -> None:
        model = _FakeModel([AIMessage(content="The answer is 4.")])
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=1, require_structured_final=False),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "The answer is 4.")
        self.assertEqual(result.stop_reason, "finalized")

    def test_rejects_multiple_tool_calls_in_one_turn(self) -> None:
        calls: list[dict[str, Any]] = []
        model = _FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 2 + 2"}, "id": "call_1"},
                        {"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_2"},
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_3"}]),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool(calls)],
            config=ControllerConfig(max_steps=2),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(calls, [])
        self.assertEqual(model.invocations[1][-2].status, "error")
        self.assertEqual(model.invocations[1][-1].status, "error")
        self.assertIn("one tool at a time", model.invocations[1][-1].content)

    def test_rejects_finalization_until_successful_sage_exec_when_configured(self) -> None:
        code = "RESULT = 2 + 2"
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_1"}]),
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": code}, "id": "call_2"}]),
                AIMessage(content="", tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_3"}]),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=3, require_successful_tool_call_for_final=True),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(model.invocations[1][-1].status, "error")
        self.assertIn("Execute sage_exec successfully", model.invocations[1][-1].content)

    def test_rejects_finalization_until_verification_passes_when_configured(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}]),
                AIMessage(content="", tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_2"}]),
                AIMessage(content="", tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_3"}]),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool(verification_summary="fail")],
            config=ControllerConfig(max_steps=3, require_verification_for_final=True),
        )

        result = controller.solve("Q")

        self.assertEqual(result.stop_reason, "max_steps_reached")
        self.assertEqual(result.verified_sage_code, "")

    def test_accepts_verified_finalization(self) -> None:
        code = "RESULT = {'verification': {'summary': 'pass'}}"
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": code}, "id": "call_1"}]),
                AIMessage(content="", tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_2"}]),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool(verification_summary="pass")],
            config=ControllerConfig(max_steps=2, require_verification_for_final=True),
        )

        result = controller.solve("Q")

        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(result.verified_sage_code, code)

    def test_tool_call_limit_returns_stable_stop_reason(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 1"}, "id": "call_1"}]),
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 2"}, "id": "call_2"}]),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=2, max_tool_calls=1),
        )

        result = controller.solve("Q")

        self.assertEqual(result.stop_reason, "max_tool_calls_reached")
        self.assertEqual(len(result.tool_traces), 1)

    def test_logger_records_messages_tool_payloads_and_tokens(self) -> None:
        code = "RESULT = 2 + 2"
        model = _FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": code}, "id": "call_1"}],
                    usage_metadata={"input_tokens": 31, "output_tokens": 9, "total_tokens": 40},
                ),
                AIMessage(
                    content="",
                    tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_2"}],
                    usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
                ),
            ]
        )
        logger = _RecordingLogger()
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool(verification_summary="pass")],
            config=ControllerConfig(max_steps=2, progress_logs=True),
            logger=logger,
        )

        result = controller.solve("Q")

        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(logger.run_metadata["question"], "Q")
        model_events = [event for event in logger.events if event["kind"] == "model_call"]
        self.assertEqual(model_events[0]["payload"]["messages"][0]["content"], "Q")
        self.assertEqual(model_events[0]["payload"]["parsed_payload"]["tool_calls"][0]["name"], SAGE_EXEC_TOOL_NAME)
        self.assertEqual(logger.token_usage_totals["input_tokens"], 43)
        self.assertEqual(logger.token_usage_totals["output_tokens"], 13)
        self.assertEqual(logger.token_usage_totals["total_tokens"], 56)

        tool_call_events = [event for event in logger.events if event["kind"] == "tool_call"]
        self.assertEqual(tool_call_events[0]["payload"]["arguments"]["code"], code)
        self.assertTrue(any("model reply:" in message for message in logger.progress_messages))
        self.assertTrue(any("tool call: sage_exec" in message for message in logger.progress_messages))
        self.assertTrue(any("tool result: sage_exec" in message for message in logger.progress_messages))

    def test_initial_message_is_user_question_without_system_prompt(self) -> None:
        model = _FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_1"}],
                )
            ]
        )
        controller = AgentController(model=model, tools=[_make_sage_tool()])

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(len(model.invocations[0]), 1)
        self.assertEqual(model.invocations[0][0].content, "Q")

    def test_configured_system_prompt_is_first_message(self) -> None:
        model = _FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4"}, "id": "call_1"}],
                )
            ]
        )
        controller = AgentController(model=model, tools=[_make_sage_tool()], system_prompt="Use Sage carefully.")

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(len(model.invocations[0]), 2)
        self.assertEqual(model.invocations[0][0].content, "Use Sage carefully.")
        self.assertEqual(model.invocations[0][1].content, "Q")


if __name__ == "__main__":
    unittest.main()
