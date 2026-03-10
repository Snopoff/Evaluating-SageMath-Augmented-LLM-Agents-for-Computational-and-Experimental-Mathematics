import types
import unittest

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.tools.catalog import make_sage_exec_tool  # noqa: E402


class SageExecToolFactoryTests(unittest.TestCase):
    def test_tool_definition_exposes_expected_spec(self) -> None:
        runtime = types.SimpleNamespace()
        tool = make_sage_exec_tool(runtime)

        self.assertEqual(tool.spec.name, "sage_exec")
        self.assertEqual(tool.spec.description, "Execute raw Sage code inside Docker.")
        self.assertEqual(tool.spec.input_schema["required"], ["code"])

    def test_handler_rejects_missing_code(self) -> None:
        runtime = types.SimpleNamespace()
        tool = make_sage_exec_tool(runtime)

        result = tool.handler({})

        self.assertFalse(result.ok)
        self.assertIn("non-empty string", result.content)

    def test_handler_maps_successful_runtime_result(self) -> None:
        runtime = types.SimpleNamespace(
            execute_sage_code=lambda **_: types.SimpleNamespace(
                status="ok",
                result_plain="4",
                result_latex="4",
                result_data={"verified": True},
                runtime_ms=12,
                stdout="",
                stderr="",
                error="",
            )
        )
        tool = make_sage_exec_tool(runtime)

        result = tool.handler({"code": "RESULT = 2 + 2", "timeout_sec": 3})

        self.assertTrue(result.ok)
        self.assertEqual(result.content, "4")
        self.assertEqual(result.metadata["status"], "ok")
        self.assertEqual(result.metadata["result_latex"], "4")
        self.assertEqual(result.metadata["result_data"], {"verified": True})

    def test_handler_maps_runtime_failure(self) -> None:
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
            )
        )
        tool = make_sage_exec_tool(runtime)

        result = tool.handler({"code": "sleep(10)"})

        self.assertFalse(result.ok)
        self.assertEqual(result.content, "Execution timed out.")
        self.assertEqual(result.metadata["status"], "timeout")


if __name__ == "__main__":
    unittest.main()
