from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmxm2.agent.controller import AgentController, ControllerConfig, ToolBudget


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


class _FakeToolClient:
    def __init__(self, response: dict[str, object]):
        self.response = response
        self.calls: list[dict[str, object]] = []

    def sage_eval(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        return dict(self.response)


class AgentControllerTests(unittest.TestCase):
    def test_tool_loop_then_finalize(self) -> None:
        client = _FakeClient(
            [
                '{"needs_tool": true, "draft_answer": "Candidate", "tool_request": {"operation": "factor", "args": {"positional_args": ["x^2-1"], "keyword_args": {}, "coerce_symbolic_strings": true}, "assumptions": {"domain": "QQ"}, "request_id": "abc", "budget_profile": "conservative"}}',
                '{"needs_tool": false, "draft_answer": "4", "tool_request": null}',
            ]
        )
        tool_client = _FakeToolClient(
            {
                "status": "ok",
                "result_plain": "4",
                "result_latex": "4",
                "runtime_ms": 5,
                "error_code": "NONE",
                "complexity_report": {"features": {}, "policy_decision": "allow", "reason": "ok"},
            }
        )

        controller = AgentController(client=client, model_name="fake", tool_client=tool_client)
        result = controller.solve("What is 2+2?")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(len(tool_client.calls), 1)
        self.assertEqual(len(result.tool_traces), 1)

    def test_non_json_output_returns_raw_text(self) -> None:
        client = _FakeClient(["The answer is approximately 10."])
        tool_client = _FakeToolClient({})
        cfg = ControllerConfig(max_turns=1, temperature=0.0)

        controller = AgentController(client=client, model_name="fake", tool_client=tool_client, config=cfg)
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "The answer is approximately 10.")
        self.assertEqual(result.stop_reason, "non_json_model_output")
        self.assertEqual(len(tool_client.calls), 0)

    def test_multi_json_output_parses_first_object(self) -> None:
        content = (
            '{"needs_tool": false, "draft_answer": "ok", "tool_request": null}\n'
            '{"needs_tool": true, "draft_answer": "ignored", "tool_request": null}'
        )
        client = _FakeClient([content])
        tool_client = _FakeToolClient({})
        controller = AgentController(client=client, model_name="fake", tool_client=tool_client)
        result = controller.solve("Q")
        self.assertEqual(result.final_answer, "ok")
        self.assertEqual(result.stop_reason, "finalized")

    def test_budget_exhausted_skips_tool_call(self) -> None:
        client = _FakeClient(
            [
                '{"needs_tool": true, "draft_answer": "draft", "tool_request": {"operation": "factor", "args": {"positional_args": ["x^2-1"], "keyword_args": {}}, "assumptions": {"domain": "QQ"}, "request_id": "abc", "budget_profile": "conservative"}}',
                '{"needs_tool": false, "draft_answer": "fallback", "tool_request": null}',
            ]
        )
        tool_client = _FakeToolClient({"status": "ok", "runtime_ms": 5})
        cfg = ControllerConfig(
            max_turns=2,
            temperature=0.0,
            tool_budget=ToolBudget(max_calls=0, max_cumulative_cpu_seconds=1, max_cumulative_wall_seconds=1),
        )

        controller = AgentController(client=client, model_name="fake", tool_client=tool_client, config=cfg)
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "fallback")
        self.assertEqual(result.stop_reason, "finalized")
        self.assertEqual(len(tool_client.calls), 0)

    def test_required_tool_mode_prompts_for_tool(self) -> None:
        client = _FakeClient(
            [
                '{"needs_tool": false, "draft_answer": "initial", "tool_request": null}',
                '{"needs_tool": true, "draft_answer": "verifying", "tool_request": {"operation": "factor", "args": {"positional_args": ["x^2-1"], "keyword_args": {}}, "assumptions": {"domain": "QQ"}, "request_id": "abc", "budget_profile": "conservative"}}',
                '{"needs_tool": false, "draft_answer": "4", "tool_request": null}',
            ]
        )
        tool_client = _FakeToolClient(
            {
                "status": "ok",
                "result_plain": "4",
                "result_latex": "4",
                "runtime_ms": 5,
                "error_code": "NONE",
                "complexity_report": {"features": {}, "policy_decision": "allow", "reason": "ok"},
            }
        )
        cfg = ControllerConfig(max_turns=3, temperature=0.0, tool_use_mode="required")

        controller = AgentController(client=client, model_name="fake", tool_client=tool_client, config=cfg)
        result = controller.solve("What is 2+2?")

        self.assertEqual(result.final_answer, "4")
        self.assertEqual(len(tool_client.calls), 1)

    def test_required_mode_with_min_calls_enforces_multiple_tool_uses(self) -> None:
        client = _FakeClient(
            [
                '{"needs_tool": false, "draft_answer": "initial", "tool_request": null}',
                '{"needs_tool": true, "draft_answer": "probe1", "tool_request": {"operation": "factor", "args": {"positional_args": ["x^2-1"], "keyword_args": {}}, "assumptions": {"domain": "QQ"}, "request_id": "a1", "budget_profile": "conservative"}}',
                '{"needs_tool": false, "draft_answer": "still early", "tool_request": null}',
                '{"needs_tool": true, "draft_answer": "probe2", "tool_request": {"operation": "factor", "args": {"positional_args": ["x^3-1"], "keyword_args": {}}, "assumptions": {"domain": "QQ"}, "request_id": "a2", "budget_profile": "conservative"}}',
                '{"needs_tool": false, "draft_answer": "final", "tool_request": null}',
            ]
        )
        tool_client = _FakeToolClient(
            {
                "status": "ok",
                "result_plain": "ok",
                "result_latex": "ok",
                "runtime_ms": 5,
                "error_code": "NONE",
                "complexity_report": {"features": {}, "policy_decision": "allow", "reason": "ok"},
            }
        )
        cfg = ControllerConfig(
            max_turns=6,
            temperature=0.0,
            tool_use_mode="required",
            min_required_tool_calls=2,
        )

        controller = AgentController(client=client, model_name="fake", tool_client=tool_client, config=cfg)
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "final")
        self.assertEqual(len(tool_client.calls), 2)

    def test_normalizes_request_defaults(self) -> None:
        client = _FakeClient(
            [
                (
                    '{"needs_tool": true, "draft_answer": "probe", "tool_request": '
                    '{"operation": "PolynomialRing", '
                    '"args": {"args": ["QQ", ["x"]], "kwargs": {"order": "lex"}}, '
                    '"assumptions": "none", "request_id": "", "budget_profile": ""}}'
                ),
                '{"needs_tool": false, "draft_answer": "done", "tool_request": null}',
            ]
        )
        tool_client = _FakeToolClient(
            {
                "status": "ok",
                "result_plain": "{}",
                "result_latex": "{}",
                "runtime_ms": 1,
                "error_code": "NONE",
                "complexity_report": {"features": {}, "policy_decision": "allow", "reason": "ok"},
            }
        )

        controller = AgentController(client=client, model_name="fake", tool_client=tool_client)
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "done")
        self.assertEqual(len(tool_client.calls), 1)
        payload = tool_client.calls[0]
        args = payload["args"]
        self.assertEqual(args["positional_args"], ["QQ", ["x"]])
        self.assertEqual(args["keyword_args"], {"order": "lex"})
        self.assertEqual(payload["assumptions"], {})
        self.assertTrue(isinstance(payload["request_id"], str) and payload["request_id"].startswith("auto_"))
        self.assertEqual(payload["budget_profile"], "conservative")

    def test_normalizes_sage_snippet_script_alias(self) -> None:
        client = _FakeClient(
            [
                (
                    '{"needs_tool": true, "draft_answer": "probe", "tool_request": '
                    '{"operation": "sage_snippet", "args": {"script": "RESULT=2+2", "result": "RESULT"}, '
                    '"assumptions": {}, "request_id": "r-snippet", "budget_profile": "small"}}'
                ),
                '{"needs_tool": false, "draft_answer": "done", "tool_request": null}',
            ]
        )
        tool_client = _FakeToolClient(
            {
                "status": "ok",
                "result_plain": "{'result_var': 'RESULT', 'result_repr': '4', 'stdout': ''}",
                "result_latex": "{}",
                "runtime_ms": 1,
                "error_code": "NONE",
                "complexity_report": {"features": {}, "policy_decision": "allow", "reason": "ok"},
            }
        )

        controller = AgentController(client=client, model_name="fake", tool_client=tool_client)
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "done")
        self.assertEqual(len(tool_client.calls), 1)
        args = tool_client.calls[0]["args"]
        self.assertEqual(args["code"], "RESULT=2+2")
        self.assertEqual(args["result_var"], "RESULT")

    def test_invalid_sage_snippet_request_is_retried_without_tool_call(self) -> None:
        client = _FakeClient(
            [
                (
                    '{"needs_tool": true, "draft_answer": "probe", "tool_request": '
                    '{"operation": "sage_snippet", "args": {"code": "<<omitted:code>>"}, '
                    '"assumptions": {}, "request_id": "r0", "budget_profile": "small"}}'
                ),
                (
                    '{"needs_tool": true, "draft_answer": "fix", "tool_request": '
                    '{"operation": "factor", "args": {"positional_args": ["x^2-1"], "keyword_args": {}}, '
                    '"assumptions": {}, "request_id": "r1", "budget_profile": "small"}}'
                ),
                '{"needs_tool": false, "draft_answer": "done", "tool_request": null}',
            ]
        )
        tool_client = _FakeToolClient(
            {
                "status": "ok",
                "result_plain": "4",
                "result_latex": "4",
                "runtime_ms": 1,
                "error_code": "NONE",
                "complexity_report": {"features": {}, "policy_decision": "allow", "reason": "ok"},
            }
        )

        controller = AgentController(client=client, model_name="fake", tool_client=tool_client)
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "done")
        self.assertEqual(len(tool_client.calls), 1)
        self.assertEqual(tool_client.calls[0]["operation"], "factor")

    def test_normalizes_generic_operation_args_aliases(self) -> None:
        client = _FakeClient(
            [
                (
                    '{"needs_tool": true, "draft_answer": "probe", "tool_request": '
                    '{"operation": "PolynomialRing", "args": {"args": ["QQ", ["x", "y"]], "kwargs": {"order": "lex"}}, '
                    '"assumptions": {}, "request_id": "r-generic", "budget_profile": "small"}}'
                ),
                '{"needs_tool": false, "draft_answer": "done", "tool_request": null}',
            ]
        )
        tool_client = _FakeToolClient(
            {
                "status": "ok",
                "result_plain": "R",
                "result_latex": "R",
                "runtime_ms": 1,
                "error_code": "NONE",
                "complexity_report": {"features": {}, "policy_decision": "allow", "reason": "ok"},
            }
        )

        controller = AgentController(client=client, model_name="fake", tool_client=tool_client)
        result = controller.solve("Q")

        self.assertEqual(result.final_answer, "done")
        self.assertEqual(len(tool_client.calls), 1)
        args = tool_client.calls[0]["args"]
        self.assertEqual(args["positional_args"], ["QQ", ["x", "y"]])
        self.assertEqual(args["keyword_args"], {"order": "lex"})


if __name__ == "__main__":
    unittest.main()
