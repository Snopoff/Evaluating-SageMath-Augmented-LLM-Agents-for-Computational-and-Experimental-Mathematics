import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import hydra.utils as hu
import rootutils
from hydra.errors import InstantiationException
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.utils.console_logging import ConsoleLogger  # noqa: E402
from src.utils.wandb_logging import WandbWeaveLogger  # noqa: E402


class _FakeConfig(dict):
    def update(self, values, allow_val_change=False):  # noqa: ARG002
        super().update(values)


class _FakeArtifact:
    def __init__(self, name: str, type: str, metadata=None) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.files: list[str] = []

    def add_file(self, path: str) -> None:
        self.files.append(path)


class _FakeRun:
    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.summary: dict[str, object] = {}
        self.logged_artifacts: list[_FakeArtifact] = []
        self.history: list[dict[str, object]] = []
        self.finished = False

    def log_artifact(self, artifact: _FakeArtifact) -> None:
        self.logged_artifacts.append(artifact)

    def log(self, payload: dict[str, object], step: int | None = None) -> None:
        record = dict(payload)
        if step is not None:
            record["_step"] = step
        self.history.append(record)

    def finish(self) -> None:
        self.finished = True


class _FakeWandbModule:
    Artifact = _FakeArtifact

    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.runs: list[_FakeRun] = []

    def init(self, **kwargs: object) -> _FakeRun:
        self.init_calls.append(dict(kwargs))
        run = _FakeRun()
        self.runs.append(run)
        return run


class _FakeWeaveModule:
    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.trace_calls: list[dict[str, object]] = []

    def init(self, project: str, global_attributes=None):
        self.init_calls.append({"project": project, "global_attributes": dict(global_attributes or {})})
        return object()

    def op(self, name: str):
        def _decorator(func):
            def _wrapped(payload):
                result = func(payload)
                self.trace_calls.append({"name": name, "payload": dict(payload)})
                return result

            return _wrapped

        return _decorator


class ExperimentLoggerTests(unittest.TestCase):
    def test_console_logger_keeps_per_agent_results_for_sequential_agents(self) -> None:
        logger = ConsoleLogger(mode="chat")

        logger.start_run(metadata={"agent_id": "single_agent", "question": "Q"})
        logger.log_solve_result(
            agent_id="single_agent",
            final_answer="4",
            turn_count=1,
            stop_reason="finalized",
            tool_traces=[],
        )

        logger.start_run(metadata={"agent_id": "review_agent", "question": "Q"})
        logger.log_solve_result(
            agent_id="review_agent",
            final_answer="confirmed",
            turn_count=1,
            stop_reason="finalized",
            tool_traces=[],
        )

        self.assertEqual(logger.run_metadata["agent_ids"], ["single_agent", "review_agent"])
        self.assertEqual(logger.final_results["single_agent"]["final_answer"], "4")
        self.assertEqual(logger.final_results["review_agent"]["final_answer"], "confirmed")
        self.assertEqual(logger.final_result["agent_id"], "review_agent")

    def test_hydra_instantiation_fails_fast_when_sdk_import_fails(self) -> None:
        cfg = OmegaConf.create(
            {
                "_target_": "src.utils.wandb_logging.WandbWeaveLogger",
                "entity": "entity",
                "project": "project",
            }
        )

        with patch("src.utils.wandb_logging.importlib.import_module", side_effect=ModuleNotFoundError("wandb")):
            with self.assertRaises(InstantiationException):
                hu.instantiate(cfg, mode="chat")

    def test_wandb_logger_uploads_verified_code_and_trace_artifacts(self) -> None:
        fake_wandb = _FakeWandbModule()
        fake_weave = _FakeWeaveModule()

        def _fake_import(name: str):
            if name == "wandb":
                return fake_wandb
            if name == "weave":
                return fake_weave
            raise ModuleNotFoundError(name)

        with patch("src.utils.wandb_logging.importlib.import_module", side_effect=_fake_import):
            logger = WandbWeaveLogger(entity="entity", project="project", mode="chat")
            logger.start_run(
                metadata={
                    "agent_id": "single_agent",
                    "question": "Q",
                    "system_prompt": "S",
                    "model_name": "fake",
                    "controller_config": {"max_steps": 2},
                    "tool_specs": [{"name": "sage_exec"}],
                }
            )
            logger.log_solve_result(
                agent_id="single_agent",
                final_answer="4",
                turn_count=2,
                stop_reason="finalized",
                tool_traces=[],
                verified_sage_code="RESULT = 4",
            )
            logger.log_model_call(
                agent_id="single_agent",
                turn=1,
                model_name="fake",
                messages=[{"role": "user", "content": "Q"}],
                raw_response="A",
                parsed_payload={"answer": "A", "tool_call": None},
                token_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )

            with tempfile.TemporaryDirectory() as tmp_dir:
                code_path = Path(tmp_dir) / "verified.py"
                code_path.write_text("RESULT = 4\n", encoding="utf-8")
                logger.log_artifact(
                    name="verified_sage_code",
                    path=code_path,
                    artifact_type="sage-code",
                    metadata={"verified": True},
                )
                logger.finish_run(status="finalized")

        run = fake_wandb.runs[0]
        artifact_names = [artifact.name for artifact in run.logged_artifacts]
        artifact_types = [artifact.type for artifact in run.logged_artifacts]
        self.assertIn("verified_sage_code", artifact_names)
        self.assertIn("sage-code", artifact_types)
        self.assertIn("llmxcas-trace", artifact_types)
        self.assertTrue(run.finished)
        self.assertEqual(run.summary["agent_count"], 1)
        self.assertEqual(run.summary["input_tokens"], 10)
        self.assertEqual(run.summary["output_tokens"], 5)
        self.assertEqual(run.summary["total_tokens"], 15)
        self.assertEqual(run.history[0]["model_call/input_tokens"], 10)
        self.assertEqual(run.history[0]["model_call/output_tokens"], 5)
        self.assertEqual(run.history[0]["model_call/total_tokens"], 15)
        self.assertEqual(len(fake_weave.init_calls), 1)
        self.assertTrue(any(call["name"] == "llmxcas_run_start" for call in fake_weave.trace_calls))
        self.assertTrue(any(call["name"] == "llmxcas_model_call" for call in fake_weave.trace_calls))
        self.assertTrue(any(call["name"] == "llmxcas_solve_result" for call in fake_weave.trace_calls))
        self.assertTrue(any(call["name"] == "llmxcas_artifact" for call in fake_weave.trace_calls))


if __name__ == "__main__":
    unittest.main()
