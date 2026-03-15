import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)


from src.sage.runtime import CONTAINER_SOURCE_FILE, SAGE_RUNNER_CODE, SageRuntime  # noqa: E402
from src.sage.types import SageRuntimeConfig  # noqa: E402


class SageRuntimeTests(unittest.TestCase):
    @staticmethod
    def _load_to_result_data():
        start = SAGE_RUNNER_CODE.index("def to_result_data")
        end = SAGE_RUNNER_CODE.index("\n\ndef main():")
        helper_source = SAGE_RUNNER_CODE[start:end]
        namespace = {"json": json}
        exec(helper_source, namespace, namespace)
        return namespace["to_result_data"]

    def test_parses_successful_runner_output(self) -> None:
        runtime = SageRuntime(SageRuntimeConfig(image="docker.io/sagemath/sagemath:latest"))
        output = json.dumps(
            {
                "status": "ok",
                "result_plain": "4",
                "result_latex": "4",
                "result_data": {"verified": True},
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
        self.assertEqual(result.result_data, {"verified": True})
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

        cmd = runtime._build_docker_cmd("{}", Path("/tmp/llmxm2_sage_exec.py"))
        self.assertIn("--platform", cmd)
        idx = cmd.index("--platform")
        self.assertEqual(cmd[idx + 1], "linux/amd64")

    def test_runtime_exec_args_pass_payload_as_argv(self) -> None:
        runtime = SageRuntime(SageRuntimeConfig(image="docker.io/sagemath/sagemath:latest"))
        payload = '{"source_file":"/tmp/llmxm2_sage_exec.py","result_var":"RESULT"}'

        args = runtime._runtime_exec_args(payload)

        self.assertEqual(args[-1], payload)
        self.assertIn(SAGE_RUNNER_CODE, args)

    def test_execute_sage_code_mounts_saved_py_artifact_and_uses_wrapper(self) -> None:
        runtime = SageRuntime(SageRuntimeConfig(image="docker.io/sagemath/sagemath:latest"))
        observed_code: list[str] = []
        observed_mount: list[str] = []

        def _fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            del kwargs
            mount_spec = cmd[cmd.index("--mount") + 1]
            observed_mount.append(mount_spec)
            source_path = mount_spec.split("src=", 1)[1].split(",dst=", 1)[0]
            with open(source_path, "r", encoding="utf-8") as handle:
                observed_code.append(handle.read())
            payload = json.loads(cmd[-1])
            self.assertEqual(payload["source_file"], CONTAINER_SOURCE_FILE)
            self.assertIn(SAGE_RUNNER_CODE, cmd)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout='LLM_SAGE_RESULT:{"status":"ok","result_plain":"4","result_latex":"4","result_data":{"verified":true},"stdout":""}\n',
                stderr="",
            )

        with patch("subprocess.run", side_effect=_fake_run):
            result = runtime.execute_sage_code("RESULT=2+2")

        self.assertEqual(result.status, "ok")
        self.assertEqual(observed_code, ["RESULT=2+2"])
        self.assertEqual(result.result_data, {"verified": True})
        self.assertTrue(observed_mount[0].endswith(f"dst={CONTAINER_SOURCE_FILE},readonly"))

    def test_runner_output_defaults_result_data_to_none_when_omitted(self) -> None:
        runtime = SageRuntime(SageRuntimeConfig(image="docker.io/sagemath/sagemath:latest"))

        with patch("subprocess.run") as mocked_run:
            mocked_run.return_value = subprocess.CompletedProcess(
                args=["docker"],
                returncode=0,
                stdout='LLM_SAGE_RESULT:{"status":"ok","result_plain":"4","result_latex":"4","stdout":""}\n',
                stderr="",
            )
            result = runtime.execute_sage_code("RESULT=2+2")

        self.assertIsNone(result.result_data)

    def test_runner_helper_preserves_plain_json_serializable_structure(self) -> None:
        to_result_data = self._load_to_result_data()

        result = to_result_data({"verified": True, "value": 7, "items": [1, "a", False]})

        self.assertEqual(result, {"verified": True, "value": 7, "items": [1, "a", False]})

    def test_runner_helper_normalizes_standard_containers_and_stringifies_unsupported_leaves(self) -> None:
        to_result_data = self._load_to_result_data()

        class _Leaf:
            def __str__(self) -> str:
                return "leaf-object"

        result = to_result_data({2: (1, {"x": _Leaf()}), "items": [None, False]})

        self.assertEqual(result, {"2": [1, {"x": "leaf-object"}], "items": [None, False]})

    def test_runner_is_raw_code_only_and_uses_sage_globals(self) -> None:
        self.assertIn("namespace = dict(globals())", SAGE_RUNNER_CODE)
        self.assertIn("to_result_data", SAGE_RUNNER_CODE)
        self.assertIn('"result_data": to_result_data(result_obj)', SAGE_RUNNER_CODE)
        self.assertNotIn('mode = payload.get("mode"', SAGE_RUNNER_CODE)
        self.assertNotIn('if mode == "operation"', SAGE_RUNNER_CODE)
        self.assertNotIn("LLMXM2_PAYLOAD", SAGE_RUNNER_CODE)


if __name__ == "__main__":
    unittest.main()
