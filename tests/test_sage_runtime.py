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

from llmxm2.sage.runtime import SageRuntime
from llmxm2.sage.types import SageRuntimeConfig


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


if __name__ == "__main__":
    unittest.main()
