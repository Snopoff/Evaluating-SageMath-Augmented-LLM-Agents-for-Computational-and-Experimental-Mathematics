import json
import tempfile
import unittest
from pathlib import Path

import hydra.utils as hu
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.agent.controller import SolveResult  # noqa: E402
from src.benchmark.generate_predictions import GeneratePredictionsConfig, GeneratePredictionsRunner  # noqa: E402
from src.utils.console_logging import ConsoleLogger  # noqa: E402


class _FakeController:
    def __init__(self, logger: ConsoleLogger, *, fail_attempts: int = 0):
        self.logger = logger
        self.fail_attempts = fail_attempts
        self.solve_calls = 0

    def solve(self, question: str) -> SolveResult:
        self.solve_calls += 1
        self.logger.start_run(metadata={"question": question, "agent_id": "fake-agent"})
        if self.solve_calls <= self.fail_attempts:
            raise TimeoutError("Request timed out.")

        return SolveResult(
            final_answer=f"answer for {question}",
            explanation="direct answer",
            confidence=4,
            verified_claims=[],
            tool_traces=[],
            turn_count=1,
            stop_reason="finalized",
            token_usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            final_payload={"final_answer": f"answer for {question}", "explanation": "direct answer", "confidence": 4},
        )


class GeneratePredictionsRunnerTests(unittest.TestCase):
    def test_config_coerces_yaml_paths(self) -> None:
        config = GeneratePredictionsConfig(
            dataset_path="data.json",
            output_dir="outputs",
        )

        self.assertEqual(config.dataset_path, Path("data.json"))
        self.assertEqual(config.output_dir, Path("outputs"))

    def test_smoke_writes_prediction_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {
                            "id": "problem-1",
                            "question": "What is 2+2?",
                            "answer": "4",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            logger = ConsoleLogger(mode="generate_predictions")
            config = GeneratePredictionsConfig(
                dataset_path=dataset_path,
                output_dir=tmp_path,
                limit=1,
            )
            runner = GeneratePredictionsRunner(
                controller=_FakeController(logger),
                config=config,
                logger=logger,
            )

            summary = runner.run()

            self.assertEqual(summary["rows"], 1)
            prediction_path = Path(summary["predictions_file"])
            summary_path = Path(summary["summary_file"])
            self.assertTrue(prediction_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertRegex(prediction_path.name, r"^predictions_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.jsonl$")
            self.assertRegex(summary_path.name, r"^summary_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.json$")

            row = json.loads(prediction_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["id"], "problem-1")
            self.assertEqual(row["question"], "What is 2+2?")
            self.assertEqual(row["ground_truth"], "4")
            self.assertEqual(row["model_final_answer"], "answer for What is 2+2?")
            self.assertEqual(row["explanation"], "direct answer")
            self.assertEqual(row["confidence"], 4)
            self.assertEqual(row["verified_claims"], [])
            self.assertEqual(row["tool_traces"], [])
            self.assertEqual(row["stop_reason"], "finalized")
            self.assertEqual(row["turn_count"], 1)
            self.assertEqual(row["token_usage"], {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
            self.assertIsInstance(row["solve_time_sec"], float)
            self.assertGreaterEqual(row["solve_time_sec"], 0.0)

    def test_separate_logger_runs_preserve_per_row_token_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {"id": "problem-1", "question": "What is 2+2?", "answer": "4"},
                        {"id": "problem-2", "question": "What is 3+3?", "answer": "6"},
                    ]
                ),
                encoding="utf-8",
            )

            logger = ConsoleLogger(mode="generate_predictions")
            config = GeneratePredictionsConfig(
                dataset_path=dataset_path,
                output_dir=tmp_path,
                limit=-1,
                separate_logger_runs=True,
            )
            runner = GeneratePredictionsRunner(
                controller=_FakeController(logger),
                config=config,
                logger=logger,
            )

            summary = runner.run()

            self.assertEqual(summary["rows"], 2)
            rows = [
                json.loads(line)
                for line in Path(summary["predictions_file"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["token_usage"], {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
            self.assertEqual(rows[1]["token_usage"], {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})
            self.assertFalse(logger.run_active)

    def test_retryable_timeout_retries_problem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.json"
            dataset_path.write_text(
                json.dumps([{"id": "problem-1", "question": "What is 2+2?", "answer": "4"}]),
                encoding="utf-8",
            )

            logger = ConsoleLogger(mode="generate_predictions")
            controller = _FakeController(logger, fail_attempts=1)
            config = GeneratePredictionsConfig(
                dataset_path=dataset_path,
                output_dir=tmp_path,
                limit=1,
                max_attempts_per_problem=2,
                retry_backoff_sec=0.0,
                sleep_sec_between_problems=0.0,
            )
            runner = GeneratePredictionsRunner(
                controller=controller,
                config=config,
                logger=logger,
            )

            summary = runner.run()

            self.assertEqual(summary["rows"], 1)
            self.assertEqual(controller.solve_calls, 2)
            prediction_path = Path(summary["predictions_file"])
            rows = [
                json.loads(line)
                for line in prediction_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["token_usage"], {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})

    def test_problem_error_is_recorded_and_batch_continues(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {"id": "problem-1", "question": "What is 2+2?", "answer": "4"},
                        {"id": "problem-2", "question": "What is 3+3?", "answer": "6"},
                    ]
                ),
                encoding="utf-8",
            )

            logger = ConsoleLogger(mode="generate_predictions")
            controller = _FakeController(logger, fail_attempts=10)
            config = GeneratePredictionsConfig(
                dataset_path=dataset_path,
                output_dir=tmp_path,
                limit=-1,
                max_attempts_per_problem=1,
                retry_backoff_sec=0.0,
                sleep_sec_between_problems=0.0,
                continue_on_problem_error=True,
            )
            runner = GeneratePredictionsRunner(
                controller=controller,
                config=config,
                logger=logger,
            )

            summary = runner.run()

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["failed_rows"], 2)
            self.assertEqual(summary["successful_rows"], 0)
            rows = [
                json.loads(line)
                for line in Path(summary["predictions_file"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["stop_reason"], "problem_failed")
            self.assertEqual(rows[0]["error"]["type"], "TimeoutError")
            self.assertEqual(rows[1]["stop_reason"], "problem_failed")

    def test_runner_is_hydra_instantiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "problems": [
                            {
                                "id": "problem-1",
                                "question": "What is 2+2?",
                                "answer": "4",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            logger = ConsoleLogger(mode="generate_predictions")
            cfg = OmegaConf.create(
                {
                    "_target_": "src.benchmark.generate_predictions.GeneratePredictionsRunner",
                    "config": {
                        "_target_": "src.benchmark.generate_predictions.GeneratePredictionsConfig",
                        "dataset_path": str(dataset_path),
                        "output_dir": str(tmp_path),
                        "limit": 1,
                    },
                }
            )

            runner = hu.instantiate(cfg, controller=_FakeController(logger), logger=logger)
            summary = runner.run()

            self.assertEqual(summary["rows"], 1)


if __name__ == "__main__":
    unittest.main()
