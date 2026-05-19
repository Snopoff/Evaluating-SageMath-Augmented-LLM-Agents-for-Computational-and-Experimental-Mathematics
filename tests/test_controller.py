import unittest
from typing import Any, Sequence

import rootutils
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool, StructuredTool, tool

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.agent.controller import (  # noqa: E402
    AgentController,
    ControllerConfig,
    SolveResult,
)
from src.agent.schemas import FinalAnswerArgs  # noqa: E402
from src.agent.schemas import SageExecArgs  # noqa: E402
from src.tools.catalog import FINAL_ANSWER_TOOL_NAME, SAGE_EXEC_TOOL_NAME  # noqa: E402
from src.utils.console_logging import ConsoleLogger  # noqa: E402


class _FakeStructuredRunnable:
    def __init__(self, model: "_FakeModel") -> None:
        self.model = model

    def invoke(self, messages: list[Any]) -> Any:
        self.model.structured_invocations.append(list(messages))
        if not self.model.structured_outputs:
            raise RuntimeError("No more fake structured outputs available")
        entry = self.model.structured_outputs.pop(0)
        if isinstance(entry, dict) and ("parsed" in entry or "parsing_error" in entry or "raw" in entry):
            raw = entry.get(
                "raw",
                AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
                ),
            )
            return {
                "raw": raw,
                "parsed": entry.get("parsed"),
                "parsing_error": entry.get("parsing_error"),
            }

        raw = AIMessage(
            content="",
            usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        )
        return {"raw": raw, "parsed": entry, "parsing_error": None}


class _FakeModel:
    def __init__(self, outputs: list[AIMessage] | None = None, structured_outputs: list[Any] | None = None) -> None:
        self.outputs = list(outputs or [])
        self.structured_outputs = list(structured_outputs or [])
        self.bound_tools: list[BaseTool] = []
        self.bind_kwargs: dict[str, Any] = {}
        self.invocations: list[list[Any]] = []
        self.structured_invocations: list[list[Any]] = []
        self.structured_kwargs: dict[str, Any] = {}
        self.model_name = "fake-model"

    def bind_tools(self, tools: Sequence[BaseTool], **kwargs: Any) -> "_FakeModel":
        self.bound_tools = list(tools)
        self.bind_kwargs = dict(kwargs)
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _FakeStructuredRunnable:  # noqa: ARG002
        self.structured_kwargs = dict(kwargs)
        if kwargs.get("include_raw") is not True:
            raise TypeError("include_raw=True is required")
        return _FakeStructuredRunnable(self)

    def invoke(self, messages: list[Any]) -> AIMessage:
        self.invocations.append(list(messages))
        if not self.outputs:
            raise RuntimeError("No more fake outputs available")
        return self.outputs.pop(0)


class _FakeModelWithoutRaw(_FakeModel):
    def with_structured_output(self, schema: Any, **kwargs: Any) -> _FakeStructuredRunnable:  # noqa: ARG002
        raise TypeError("include_raw is not supported")


class _RecordingLogger(ConsoleLogger):
    def __init__(self) -> None:
        super().__init__(mode="test")
        self.progress_messages: list[str] = []

    def progress(self, message: str) -> None:
        self.progress_messages.append(message)

    def log(self, message: str, level: str = "info", color: str = "white", *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.progress_messages.append(f"[{level}] {message}")


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


def _make_non_sage_tool() -> BaseTool:
    @tool("context7")
    def context7(query: str) -> str:
        """Lookup supporting context."""

        return query

    return context7


def _make_async_sage_tool(calls: list[dict[str, Any]] | None = None) -> BaseTool:
    async def sage_exec(code: str, result_var: str = "RESULT") -> tuple[str, dict[str, Any]]:
        if calls is not None:
            calls.append({"code": code, "result_var": result_var})
        return "4", {
            "ok": True,
            "status": "ok",
            "verification": _verification_payload(),
            "code": code,
        }

    return StructuredTool(
        name=SAGE_EXEC_TOOL_NAME,
        description="Execute Sage.",
        args_schema=SageExecArgs,
        coroutine=sage_exec,
        response_format="content_and_artifact",
    )


class AgentControllerTests(unittest.TestCase):
    def test_tool_loop_then_submit_final_answer(self) -> None:
        code = "RESULT = 2 + 2"
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": code}, "id": "call_1"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {
                                "final_answer": "4",
                                "sympy_answer": "4",
                                "explanation": "verified",
                                "confidence": 5,
                                "verified_claims": ["computed 2 + 2"],
                            },
                            "id": "call_2",
                        }
                    ],
                ),
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
        self.assertEqual(result.sympy_answer, "4")
        self.assertEqual(result.explanation, "verified")
        self.assertEqual(result.confidence, 5)
        self.assertEqual(result.verified_claims, ["computed 2 + 2"])
        self.assertEqual(result.final_payload["final_answer"], "4")
        self.assertEqual(result.final_payload["sympy_answer"], "4")
        self.assertEqual(result.final_payload["explanation"], "verified")
        self.assertEqual(result.final_payload["confidence"], 5)
        self.assertEqual(result.final_payload["verified_claims"], ["computed 2 + 2"])
        self.assertEqual(result.final_payload["sage_code"], code)
        self.assertEqual(result.token_usage, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(calls, [{"code": code}])
        self.assertEqual(len(result.tool_traces), 1)
        self.assertEqual(result.verified_sage_code, code)
        self.assertFalse(model.bind_kwargs["parallel_tool_calls"])
        self.assertEqual({tool.name for tool in model.bound_tools}, {SAGE_EXEC_TOOL_NAME, FINAL_ANSWER_TOOL_NAME})

    def test_plain_text_is_reprompted_when_structured_final_required(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 2 + 2"}, "id": "call_1"}]),
                AIMessage(content="4"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {
                                "final_answer": "4",
                                "sympy_answer": "4",
                                "explanation": "verified",
                                "confidence": 4,
                                "verified_claims": [],
                            },
                            "id": "call_2",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=3),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.confidence, 4)
        self.assertEqual(result.stop_reason, "finalized")
        self.assertIn(f"Use the {FINAL_ANSWER_TOOL_NAME} tool", model.invocations[2][-1].content)

    def test_solve_result_stores_explicit_structured_fields(self) -> None:
        solve_result = SolveResult(
            final_answer="4",
            tool_traces=[],
            turn_count=1,
            stop_reason="finalized",
            sympy_answer="4",
            explanation="Sage verified it.",
            confidence=5,
            verified_claims=["computed 2 + 2"],
            final_payload={
                "final_answer": "4",
                "sympy_answer": "4",
                "explanation": "Sage verified it.",
                "confidence": 5,
            },
        )

        self.assertEqual(solve_result.explanation, "Sage verified it.")
        self.assertEqual(solve_result.confidence, 5)
        self.assertEqual(solve_result.verified_claims, ["computed 2 + 2"])
        self.assertEqual(solve_result.sympy_answer, "4")

    def test_plain_mode_invokes_structured_output_without_binding_tools(self) -> None:
        model = _FakeModel(
            structured_outputs=[
                FinalAnswerArgs(
                    final_answer="4",
                    sympy_answer="4",
                    explanation="direct",
                    confidence=4,
                )
            ]
        )
        logger = _RecordingLogger()
        controller = AgentController(
            model=model,
            tools=[],
            config=ControllerConfig(progress_logs=True),
            logger=logger,
            system_prompt="Answer directly.",
        )

        result = controller.solve("What is 2+2?")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.sympy_answer, "4")
        self.assertEqual(result.explanation, "direct")
        self.assertEqual(result.confidence, 4)
        self.assertEqual(
            result.final_payload,
            {"final_answer": "4", "sympy_answer": "4", "explanation": "direct", "confidence": 4},
        )
        self.assertNotIn("verified_claims", result.final_payload)
        self.assertNotIn("sage_code", result.final_payload)
        self.assertEqual(result.tool_traces, [])
        self.assertEqual(result.turn_count, 1)
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(result.token_usage, {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6})
        self.assertEqual(model.bound_tools, [])
        self.assertEqual(model.bind_kwargs, {})
        self.assertEqual(len(model.invocations), 0)
        self.assertEqual(len(model.structured_invocations), 1)
        self.assertEqual(model.structured_kwargs, {"include_raw": True})
        self.assertEqual(model.structured_invocations[0][0].content, "Answer directly.")
        self.assertEqual(model.structured_invocations[0][1].content, "What is 2+2?")
        self.assertEqual(logger.run_metadata["tool_specs"], [])
        self.assertEqual(logger.run_metadata["agent_mode"], "plain")
        self.assertEqual(logger.token_usage_totals["total_tokens"], 6)

    def test_plain_mode_accepts_multi_sympy_answer(self) -> None:
        model = _FakeModel(
            structured_outputs=[
                FinalAnswerArgs(
                    final_answer="x \\in \\{-1, 1\\}",
                    sympy_answer=["-1", "1"],
                    explanation="Solve x^2 = 1 and check both roots.",
                    confidence=4,
                )
            ]
        )
        controller = AgentController(model=model, tools=[])

        result = controller.solve("Solve x^2 = 1.")

        self.assertEqual(result.final_answer, "x \\in \\{-1, 1\\}")
        self.assertEqual(result.sympy_answer, ["-1", "1"])
        self.assertEqual(result.final_payload["sympy_answer"], ["-1", "1"])

    def test_plain_mode_retries_once_after_invalid_sympy_answer(self) -> None:
        try:
            FinalAnswerArgs.model_validate(
                {
                    "final_answer": "M_{n+1} - 2M_{n-1} + M_{n-3}",
                    "sympy_answer": "M_{n-1}",
                    "explanation": "direct",
                    "confidence": 4,
                }
            )
        except Exception as exc:  # noqa: BLE001
            parsing_error = exc
        else:
            self.fail("Expected invalid sympy_answer to raise")

        model = _FakeModel(
            structured_outputs=[
                {
                    "raw": AIMessage(
                        content='{"final_answer":"M_{n+1}-2M_{n-1}+M_{n-3}","sympy_answer":"M_{n-1}","explanation":"direct","confidence":4}',
                        usage_metadata={"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
                    ),
                    "parsed": None,
                    "parsing_error": parsing_error,
                },
                FinalAnswerArgs(
                    final_answer="M_{n+1} - 2M_{n-1} + M_{n-3}",
                    sympy_answer="M_n_plus_1 - 2*M_n_minus_1 + M_n_minus_3",
                    explanation="direct",
                    confidence=4,
                ),
            ]
        )
        logger = _RecordingLogger()
        controller = AgentController(model=model, tools=[], logger=logger)

        result = controller.solve("Express the answer in terms of Motzkin numbers.")

        self.assertEqual(result.sympy_answer, "M_n_plus_1 - 2*M_n_minus_1 + M_n_minus_3")
        self.assertEqual(result.turn_count, 2)
        self.assertEqual(result.token_usage, {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
        self.assertEqual(len(model.structured_invocations), 2)
        self.assertIn("M_n_minus_1", model.structured_invocations[1][-1].content)
        self.assertIn("sympy_answer", model.structured_invocations[1][-1].content)

    def test_plain_mode_fails_after_one_invalid_sympy_retry(self) -> None:
        try:
            FinalAnswerArgs.model_validate(
                {
                    "final_answer": "M_{n+1} - 2M_{n-1} + M_{n-3}",
                    "sympy_answer": "M_{n-1}",
                    "explanation": "direct",
                    "confidence": 4,
                }
            )
        except Exception as exc:  # noqa: BLE001
            parsing_error = exc
        else:
            self.fail("Expected invalid sympy_answer to raise")

        model = _FakeModel(
            structured_outputs=[
                {"parsed": None, "parsing_error": parsing_error},
                {"parsed": None, "parsing_error": parsing_error},
            ]
        )
        controller = AgentController(model=model, tools=[])

        with self.assertRaisesRegex(ValueError, "Structured output parsing failed"):
            controller.solve("Express the answer in terms of Motzkin numbers.")

        self.assertEqual(len(model.structured_invocations), 2)

    def test_plain_mode_requires_include_raw_structured_output_support(self) -> None:
        with self.assertRaisesRegex(TypeError, "include_raw"):
            AgentController(model=_FakeModelWithoutRaw(), tools=[])

    def test_non_empty_tool_list_uses_react_loop(self) -> None:
        model = _FakeModel(
            outputs=[
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        }
                    ],
                ),
            ]
        )
        logger = _RecordingLogger()
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            logger=logger,
        )

        result = controller.solve("What is 2+2?")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.confidence, 5)
        self.assertEqual(len(result.tool_traces), 1)
        self.assertEqual(len(model.invocations), 2)
        self.assertEqual(len(model.structured_invocations), 0)
        self.assertEqual(logger.run_metadata["agent_mode"], "react")

    def test_async_only_tool_executes_via_ainvoke_fallback(self) -> None:
        calls: list[dict[str, Any]] = []
        model = _FakeModel(
            outputs=[
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_async_sage_tool(calls)],
            config=ControllerConfig(max_steps=3),
        )

        result = controller.solve("What is 2+2?")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(calls, [{"code": "RESULT = 4", "result_var": "RESULT"}])
        self.assertEqual(len(result.tool_traces), 1)
        self.assertEqual(result.tool_traces[0]["ok"], True)

    def test_non_empty_tool_list_requires_sage_exec(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires the sage_exec tool"):
            AgentController(model=_FakeModel(), tools=[_make_non_sage_tool()])

    def test_rejects_multiple_tool_calls_in_one_turn(self) -> None:
        calls: list[dict[str, Any]] = []
        model = _FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 2 + 2"}, "id": "call_1"},
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        },
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_3"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_4",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool(calls)],
            config=ControllerConfig(max_steps=3),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(calls, [{"code": "RESULT = 4"}])
        self.assertEqual(model.invocations[1][-2].status, "error")
        self.assertEqual(model.invocations[1][-1].status, "error")
        self.assertIn("one tool at a time", model.invocations[1][-1].content)

    def test_rejects_finalization_until_successful_sage_exec(self) -> None:
        code = "RESULT = 2 + 2"
        model = _FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": code}, "id": "call_2"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_3",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=3),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(model.invocations[1][-1].status, "error")
        self.assertIn("Execute sage_exec successfully", model.invocations[1][-1].content)

    def test_rejects_sage_finalization_without_verified_claims_key(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5},
                            "id": "call_2",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_3",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(model=model, tools=[_make_sage_tool()], config=ControllerConfig(max_steps=3))

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(model.invocations[2][-1].status, "error")
        self.assertIn("Invalid submit_final_answer arguments", model.invocations[2][-1].content)

    def test_rejects_finalization_without_explanation(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}]),
                AIMessage(content="", tool_calls=[{"name": FINAL_ANSWER_TOOL_NAME, "args": {"final_answer": "4", "sympy_answer": "4", "verified_claims": []}, "id": "call_2"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_3",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=3),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(model.invocations[2][-1].status, "error")
        self.assertIn("Invalid submit_final_answer arguments", model.invocations[2][-1].content)

    def test_rejects_finalization_with_latex_sympy_answer(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "$4$", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_3",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(model=model, tools=[_make_sage_tool()], config=ControllerConfig(max_steps=3))

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.sympy_answer, "4")
        self.assertEqual(model.invocations[2][-1].status, "error")
        self.assertIn("Invalid submit_final_answer arguments", model.invocations[2][-1].content)

    def test_rejects_finalization_with_non_parseable_sympy_answer(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "x =", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_3",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(model=model, tools=[_make_sage_tool()], config=ControllerConfig(max_steps=3))

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.sympy_answer, "4")
        self.assertEqual(model.invocations[2][-1].status, "error")
        self.assertIn("Invalid submit_final_answer arguments", model.invocations[2][-1].content)

    def test_rejects_finalization_until_verification_passes_when_configured(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_3",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool(verification_summary="fail")],
            config=ControllerConfig(max_steps=2, require_verification_for_final=True),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "forced_finalized_without_verification")

    def test_forces_finalization_after_step_limit(self) -> None:
        code = "RESULT = 2 + 2"
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": code}, "id": "call_1"}]),
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_2"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_3",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=2),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "forced_finalized")
        self.assertEqual(result.turn_count, 3)
        self.assertEqual(result.verified_sage_code, "RESULT = 4")
        self.assertEqual(result.final_payload["sage_code"], "RESULT = 4")
        self.assertIn("step limit", model.invocations[2][-1].content)
        self.assertIn("Do not call sage_exec again", model.invocations[2][-1].content)

    def test_forced_finalization_marks_missing_successful_sage(self) -> None:
        model = _FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {
                                "final_answer": "Best unverified answer: 4",
                                "sympy_answer": "4",
                                "explanation": "verified",
                                "confidence": 2,
                                "verified_claims": [],
                            },
                            "id": "call_2",
                        }
                    ],
                ),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=1),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "Best unverified answer: 4")
        self.assertEqual(result.stop_reason, "forced_finalized_without_successful_sage")
        self.assertEqual(result.verified_sage_code, "")

    def test_forced_finalization_fails_without_final_tool_call(self) -> None:
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 2 + 2"}, "id": "call_1"}]),
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_2"}]),
                AIMessage(content="The answer is 4."),
            ]
        )
        controller = AgentController(
            model=model,
            tools=[_make_sage_tool()],
            config=ControllerConfig(max_steps=2),
        )

        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "")
        self.assertEqual(result.stop_reason, "forced_finalization_failed")
        self.assertEqual(result.explanation, "Forced finalization failed: the model did not call submit_final_answer with valid arguments.")

    def test_accepts_verified_finalization(self) -> None:
        code = "RESULT = {'verification': {'summary': 'pass'}}"
        model = _FakeModel(
            [
                AIMessage(content="", tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": code}, "id": "call_1"}]),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        }
                    ],
                ),
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
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        }
                    ],
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
        self.assertEqual(result.token_usage, {"input_tokens": 43, "output_tokens": 13, "total_tokens": 56})
        model_events = [event for event in logger.events if event["kind"] == "model_call"]
        self.assertEqual(model_events[0]["payload"]["messages"][0]["content"], "Q")
        self.assertEqual(model_events[0]["payload"]["parsed_payload"]["tool_calls"][0]["name"], SAGE_EXEC_TOOL_NAME)
        self.assertEqual(logger.token_usage_totals["input_tokens"], 43)
        self.assertEqual(logger.token_usage_totals["output_tokens"], 13)
        self.assertEqual(logger.token_usage_totals["total_tokens"], 56)

        tool_call_events = [event for event in logger.events if event["kind"] == "tool_call"]
        self.assertEqual(tool_call_events[0]["payload"]["arguments"]["code"], code)
        self.assertTrue(any("model_reply" in message for message in logger.progress_messages))
        self.assertTrue(any("tool_call" in message and "sage_exec" in message for message in logger.progress_messages))
        self.assertTrue(any("tool_result" in message and "sage_exec" in message for message in logger.progress_messages))

    def test_token_usage_falls_back_to_response_metadata(self) -> None:
        class _ResponseMetadataStructuredRunnable:
            def invoke(self, messages: list[Any]) -> Any:
                model.structured_invocations.append(list(messages))
                parsed = model.structured_outputs.pop(0)
                raw = AIMessage(
                    content="",
                    response_metadata={"token_usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10}},
                )
                return {"raw": raw, "parsed": parsed, "parsing_error": None}

        class _ResponseMetadataModel(_FakeModel):
            def with_structured_output(self, schema: Any, **kwargs: Any) -> _ResponseMetadataStructuredRunnable:  # noqa: ARG002
                self.structured_kwargs = dict(kwargs)
                return _ResponseMetadataStructuredRunnable()

        model = _ResponseMetadataModel(
            structured_outputs=[
                FinalAnswerArgs(
                    final_answer="4",
                    sympy_answer="4",
                    explanation="direct",
                    confidence=4,
                )
            ]
        )
        controller = AgentController(model=model, tools=[])

        result = controller.solve("Q")

        self.assertEqual(result.token_usage, {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10})

    def test_initial_message_is_user_question_without_system_prompt(self) -> None:
        model = _FakeModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        }
                    ],
                ),
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
                    tool_calls=[{"name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 4"}, "id": "call_1"}],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": FINAL_ANSWER_TOOL_NAME,
                            "args": {"final_answer": "4", "sympy_answer": "4", "explanation": "verified", "confidence": 5, "verified_claims": []},
                            "id": "call_2",
                        }
                    ],
                ),
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
