import json
import types
import unittest

import rootutils
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.tools.catalog import (  # noqa: E402
    FINAL_ANSWER_TOOL_NAME,
    SAGE_EXEC_TOOL_NAME,
    make_sage_exec_tool,
    make_submit_final_answer_tool,
)


class SageExecToolFactoryTests(unittest.TestCase):
    def test_sage_exec_tool_exposes_langchain_metadata(self) -> None:
        runtime = types.SimpleNamespace()
        sage_tool = make_sage_exec_tool(runtime, usage_notes="Use RESULT.")

        self.assertEqual(sage_tool.name, SAGE_EXEC_TOOL_NAME)
        self.assertIn("Execute Sage script code", sage_tool.description)
        self.assertIn("Sage preparser syntax is allowed", sage_tool.description)
        self.assertIn("Use RESULT.", sage_tool.description)
        self.assertIn("code", sage_tool.args)
        self.assertNotIn("timeout_sec", sage_tool.args)

    def test_sage_exec_tool_rejects_missing_code_by_schema(self) -> None:
        runtime = types.SimpleNamespace()
        sage_tool = make_sage_exec_tool(runtime)

        with self.assertRaises(ValidationError):
            sage_tool.invoke({})

    def test_sage_exec_tool_returns_tool_message_with_artifact(self) -> None:
        runtime = types.SimpleNamespace(
            execute_sage_code=lambda **_: types.SimpleNamespace(
                status="ok",
                result_plain="4",
                result_latex="4",
                result_data={"verification": {"summary": "pass", "checks": [{"id": "constraint_1", "status": "pass", "evidence": "ok"}], "outputs": {}}},
                runtime_ms=12,
                stdout="",
                stderr="",
                error="",
                error_kind="",
            )
        )
        sage_tool = make_sage_exec_tool(runtime)

        result = sage_tool.invoke({"type": "tool_call", "id": "call_1", "name": SAGE_EXEC_TOOL_NAME, "args": {"code": "RESULT = 2 + 2"}})

        self.assertIsInstance(result, ToolMessage)
        self.assertEqual(result.content, "4")
        self.assertTrue(result.artifact["ok"])
        self.assertEqual(result.artifact["status"], "ok")
        self.assertEqual(result.artifact["result_latex"], "4")
        self.assertEqual(result.artifact["verification"]["summary"], "pass")
        self.assertEqual(result.artifact["code"], "RESULT = 2 + 2")

    def test_sage_exec_tool_ignores_model_supplied_timeout(self) -> None:
        calls: list[dict[str, object]] = []
        runtime = types.SimpleNamespace(
            execute_sage_code=lambda **kwargs: (
                calls.append(dict(kwargs))
                or types.SimpleNamespace(
                    status="ok",
                    result_plain="4",
                    result_latex="4",
                    result_data=None,
                    runtime_ms=12,
                    stdout="",
                    stderr="",
                    error="",
                    error_kind="",
                )
            )
        )
        sage_tool = make_sage_exec_tool(runtime)

        sage_tool.invoke(
            {
                "type": "tool_call",
                "id": "call_1",
                "name": SAGE_EXEC_TOOL_NAME,
                "args": {"code": "RESULT = 2 + 2", "timeout_sec": 120},
            }
        )

        self.assertEqual(calls, [{"code": "RESULT = 2 + 2", "result_var": "RESULT"}])

    def test_sage_exec_tool_maps_runtime_failure(self) -> None:
        runtime = types.SimpleNamespace(
            execute_sage_code=lambda **_: types.SimpleNamespace(
                status="timeout",
                result_plain="",
                result_latex="",
                result_data=None,
                runtime_ms=50,
                stdout="",
                stderr="runtime stderr",
                error="Execution timed out.",
                error_kind="timeout",
            )
        )
        sage_tool = make_sage_exec_tool(runtime)

        result = sage_tool.invoke({"type": "tool_call", "id": "call_1", "name": SAGE_EXEC_TOOL_NAME, "args": {"code": "sleep(10)"}})

        self.assertEqual(result.content, "Execution timed out.")
        self.assertFalse(result.artifact["ok"])
        self.assertEqual(result.artifact["status"], "timeout")
        self.assertEqual(result.artifact["error_kind"], "timeout")

    def test_submit_final_answer_tool_exposes_schema(self) -> None:
        final_tool = make_submit_final_answer_tool()

        self.assertEqual(final_tool.name, FINAL_ANSWER_TOOL_NAME)
        self.assertIn("final_answer", final_tool.args)
        self.assertIn("sympy_answer", final_tool.args)
        self.assertIn("explanation", final_tool.args)
        self.assertIn("confidence", final_tool.args)
        self.assertIn("verified_claims", final_tool.args)
        self.assertEqual(
            json.loads(
                final_tool.invoke(
                    {
                        "final_answer": "4",
                        "sympy_answer": "4",
                        "explanation": "verified",
                        "confidence": 5,
                        "verified_claims": ["computed"],
                    }
                )
            ),
            {
                "explanation": "verified",
                "final_answer": "4",
                "sympy_answer": "4",
                "confidence": 5,
                "verified_claims": ["computed"],
            },
        )

    def test_submit_final_answer_tool_rejects_latex_or_caret_sympy_answer(self) -> None:
        final_tool = make_submit_final_answer_tool()

        with self.assertRaises(ValidationError):
            final_tool.invoke(
                {
                    "final_answer": "x^2",
                    "sympy_answer": "x^2",
                    "explanation": "verified",
                    "confidence": 5,
                    "verified_claims": [],
                }
            )

        with self.assertRaises(ValidationError):
            final_tool.invoke(
                {
                    "final_answer": "$x$",
                    "sympy_answer": "$x$",
                    "explanation": "verified",
                    "confidence": 5,
                    "verified_claims": [],
                }
            )

    def test_submit_final_answer_tool_rejects_non_parseable_sympy_answer(self) -> None:
        final_tool = make_submit_final_answer_tool()

        with self.assertRaises(ValidationError):
            final_tool.invoke(
                {
                    "final_answer": "x =",
                    "sympy_answer": "x =",
                    "explanation": "verified",
                    "confidence": 5,
                    "verified_claims": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
