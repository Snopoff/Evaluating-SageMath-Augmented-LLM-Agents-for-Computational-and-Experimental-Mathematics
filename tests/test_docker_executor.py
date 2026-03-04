from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmxm2.mcp.docker_executor import DockerRuntimeConfig, DockerSageExecutor
from llmxm2.mcp.policy import SageEvalRequest


_VALID_IMAGE = "docker.io/sagemath/sagemath@sha256:" + "a" * 64


class DockerExecutorTests(unittest.TestCase):
    def _request(self) -> SageEvalRequest:
        return SageEvalRequest.from_payload(
            {
                "operation": "factor",
                "args": {"positional_args": ["x^2-1"], "keyword_args": {}},
                "assumptions": {"domain": "QQ"},
                "request_id": "req-1",
                "budget_profile": "conservative",
            }
        )

    def test_parses_successful_runner_output(self) -> None:
        executor = DockerSageExecutor(DockerRuntimeConfig(image=_VALID_IMAGE))
        output = json.dumps(
            {
                "status": "ok",
                "result_plain": "4",
                "result_latex": "4",
                "error_code": "NONE",
            }
        )

        with patch("subprocess.run") as mocked_run:
            mocked_run.return_value = subprocess.CompletedProcess(
                args=["docker"],
                returncode=0,
                stdout=output,
                stderr="",
            )
            result = executor.execute(self._request())

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.result_plain, "4")

    def test_timeout_maps_to_timeout_status(self) -> None:
        executor = DockerSageExecutor(DockerRuntimeConfig(image=_VALID_IMAGE))

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=8)):
            result = executor.execute(self._request())

        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.error_code, "TIMEOUT")

    def test_requires_digest_pinned_image(self) -> None:
        with self.assertRaises(ValueError):
            DockerSageExecutor(DockerRuntimeConfig(image="docker.io/sagemath/sagemath:latest"))

    def test_platform_flag_is_included_when_configured(self) -> None:
        executor = DockerSageExecutor(
            DockerRuntimeConfig(
                image=_VALID_IMAGE,
                platform="linux/amd64",
            )
        )
        cmd = executor._build_docker_cmd()
        self.assertIn("--platform", cmd)
        idx = cmd.index("--platform")
        self.assertEqual(cmd[idx + 1], "linux/amd64")

    def test_cpu_ulimit_is_included_when_configured(self) -> None:
        executor = DockerSageExecutor(
            DockerRuntimeConfig(
                image=_VALID_IMAGE,
                cpu_seconds=7.2,
            )
        )
        cmd = executor._build_docker_cmd()
        self.assertIn("--ulimit", cmd)
        idx = cmd.index("--ulimit")
        self.assertEqual(cmd[idx + 1], "cpu=8")

    def test_home_and_dot_sage_env_are_injected(self) -> None:
        executor = DockerSageExecutor(
            DockerRuntimeConfig(
                image=_VALID_IMAGE,
                home_dir="/tmp",
                dot_sage_dir="/tmp/.sage",
            )
        )
        cmd = executor._build_docker_cmd()
        self.assertIn("-e", cmd)
        self.assertIn("HOME=/tmp", cmd)
        self.assertIn("DOT_SAGE=/tmp/.sage", cmd)

    def test_entrypoint_is_included_when_configured(self) -> None:
        executor = DockerSageExecutor(
            DockerRuntimeConfig(
                image=_VALID_IMAGE,
                entrypoint="/bin/bash",
            )
        )
        cmd = executor._build_docker_cmd()
        self.assertIn("--entrypoint", cmd)
        idx = cmd.index("--entrypoint")
        self.assertEqual(cmd[idx + 1], "/bin/bash")
        self.assertIn("-lc", cmd)

    def test_user_flag_is_omitted_when_user_not_set(self) -> None:
        executor = DockerSageExecutor(
            DockerRuntimeConfig(
                image=_VALID_IMAGE,
                user="",
            )
        )
        cmd = executor._build_docker_cmd()
        self.assertNotIn("--user", cmd)

    def test_user_attempts_fallback_to_default_user(self) -> None:
        executor = DockerSageExecutor(
            DockerRuntimeConfig(
                image=_VALID_IMAGE,
                user="0:0",
            )
        )
        attempts = executor._user_attempts()
        self.assertEqual(attempts[0], "0:0")
        self.assertEqual(attempts[1], "")

    def test_from_config_normalizes_plain_sage_entrypoint_to_bash(self) -> None:
        executor = DockerSageExecutor.from_config(
            {
                "image": _VALID_IMAGE,
                "entrypoint": "sage",
            }
        )
        self.assertEqual(executor.config.entrypoint, "/bin/bash")

    def test_retryable_entrypoint_error_detection(self) -> None:
        self.assertTrue(
            DockerSageExecutor._is_retryable_entrypoint_error(
                'exec: "sage": executable file not found in $PATH'
            )
        )
        self.assertTrue(
            DockerSageExecutor._is_retryable_entrypoint_error(
                'exec: "/bin/bash": stat /bin/bash: no such file or directory'
            )
        )
        self.assertTrue(
            DockerSageExecutor._is_retryable_entrypoint_error(
                "/usr/local/bin/sage-entrypoint: line 8: exec: sage: not found"
            )
        )

    def test_entrypoint_attempts_include_default_entrypoint_fallback(self) -> None:
        executor = DockerSageExecutor(
            DockerRuntimeConfig(
                image=_VALID_IMAGE,
                entrypoint="/bin/sh",
            )
        )
        attempts = executor._entrypoint_attempts()
        self.assertEqual(attempts[0], "/bin/sh")
        self.assertIn("", attempts)

    def test_runtime_args_include_payload_argument(self) -> None:
        executor = DockerSageExecutor(DockerRuntimeConfig(image=_VALID_IMAGE))
        payload = '{"operation":"factor"}'
        args = executor._runtime_exec_args("/bin/bash", payload)
        self.assertEqual(args[-1], payload)
        self.assertIn("-lc", args)


if __name__ == "__main__":
    unittest.main()
