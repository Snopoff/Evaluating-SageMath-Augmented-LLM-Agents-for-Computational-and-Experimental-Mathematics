import json
import subprocess
import unittest
from unittest.mock import patch

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.sage.runtime import SAGE_RUNNER_CODE, SageRuntime  # noqa: E402
from src.sage.types import SageRuntimeConfig  # noqa: E402


class SageRuntimeTests(unittest.TestCase):
    def test_parses_successful_runner_output(self) -> None:
        runtime = SageRuntime(SageRuntimeConfig(image="docker.io/sagemath/sagemath:latest"))
        output = json.dumps(
            {
                "status": "ok",
                "result_plain": "4",
                "result_latex": "4",
                "stdout": "",
            }
        )

        with patch("subprocess.run") as mocked_run:
            mocked_run.return_value = subprocess.CompletedProcess(
                args=["docker"],
                returncode=0,
                stdout=output,
                stderr="",
            )
            result = runtime.execute_sage_code("RESULT=2+2")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.result_plain, "4")
        self.assertNotIn("input", mocked_run.call_args.kwargs)

    def test_timeout_maps_to_timeout_status(self) -> None:
        runtime = SageRuntime(SageRuntimeConfig(image="docker.io/sagemath/sagemath:latest"))

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=8)):
            result = runtime.execute_sage_code("RESULT=2+2")

        self.assertEqual(result.status, "timeout")

    def test_platform_flag_included_when_configured(self) -> None:
        runtime = SageRuntime(
            SageRuntimeConfig(
                image="docker.io/sagemath/sagemath:latest",
                platform="linux/amd64",
            )
        )

        cmd = runtime._build_docker_cmd("{}")
        self.assertIn("--platform", cmd)
        idx = cmd.index("--platform")
        self.assertEqual(cmd[idx + 1], "linux/amd64")

    def test_runtime_exec_args_pass_payload_as_argv(self) -> None:
        runtime = SageRuntime(SageRuntimeConfig(image="docker.io/sagemath/sagemath:latest"))

        args = runtime._runtime_exec_args('{"code":"RESULT=2+2"}')

        self.assertEqual(args[-1], '{"code":"RESULT=2+2"}')
        self.assertIn(SAGE_RUNNER_CODE, args)

    def test_runner_is_raw_code_only_and_uses_sage_globals(self) -> None:
        self.assertIn("namespace = dict(globals())", SAGE_RUNNER_CODE)
        self.assertNotIn('mode = payload.get("mode"', SAGE_RUNNER_CODE)
        self.assertNotIn('if mode == "operation"', SAGE_RUNNER_CODE)
        self.assertNotIn("LLMXM2_PAYLOAD", SAGE_RUNNER_CODE)


if __name__ == "__main__":
    unittest.main()
