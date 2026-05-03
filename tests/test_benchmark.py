import json
import tempfile
import unittest
from pathlib import Path

import hydra.utils as hu
import rootutils
from omegaconf import OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.agent.controller import SolveResult  # noqa: E402
from src.benchmark.runner import BenchmarkConfig, BenchmarkRunner  # noqa: E402


class _FakeController:
    def __init__(self, answer: str):
        self.answer = answer

    def solve(self, question: str) -> SolveResult:  # noqa: ARG002
        return SolveResult(
            final_answer=self.answer,
            tool_traces=[],
            turn_count=1,
            stop_reason="finalized",
        )


class BenchmarkRunnerTests(unittest.TestCase):
    def test_config_coerces_yaml_paths(self) -> None:
        config = BenchmarkConfig(
            dataset_path="data.jsonl",
            output_dir="outputs",
        )

        self.assertEqual(config.dataset_path, Path("data.jsonl"))
        self.assertEqual(config.output_dir, Path("outputs"))

    def test_smoke_writes_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "problem-1",
                        "question": "What is 2+2?",
                        "answer": "4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            config = BenchmarkConfig(
                dataset_path=dataset_path,
                output_dir=tmp_path,
                limit=1,
                predictions_file="predictions.jsonl",
                tool_traces_file="tool_traces.jsonl",
                metrics_file="metrics.json",
            )
            runner = BenchmarkRunner(
                controller=_FakeController("4"),
                config=config,
            )

            metrics = runner.run()

            self.assertEqual(metrics["rows"], 1)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertTrue((tmp_path / "predictions.jsonl").exists())
            self.assertTrue((tmp_path / "tool_traces.jsonl").exists())
            self.assertTrue((tmp_path / "metrics.json").exists())

    def test_reads_json_problem_collection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "benchmark_name": "sample",
                        "problems": [
                            {
                                "id": "problem-1",
                                "question": "What is 2+2?",
                                "answer": "4",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = BenchmarkConfig(
                dataset_path=dataset_path,
                output_dir=tmp_path,
                limit=1,
            )
            runner = BenchmarkRunner(
                controller=_FakeController("4"),
                config=config,
            )

            metrics = runner.run()

            self.assertEqual(metrics["rows"], 1)
            self.assertEqual(metrics["accuracy"], 1.0)

    def test_runner_is_hydra_instantiable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "problem-1",
                        "question": "What is 2+2?",
                        "answer": "4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            cfg = OmegaConf.create(
                {
                    "_target_": "src.benchmark.runner.BenchmarkRunner",
                    "config": {
                        "_target_": "src.benchmark.runner.BenchmarkConfig",
                        "dataset_path": str(dataset_path),
                        "output_dir": str(tmp_path),
                        "limit": 1,
                    },
                }
            )

            runner = hu.instantiate(cfg, controller=_FakeController("4"))
            metrics = runner.run()

            self.assertEqual(metrics["rows"], 1)
            self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
