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
    def __init__(self, final_answer: str, sympy_answer: str | list[str] | None = None):
        self.final_answer = final_answer
        self.sympy_answer = final_answer if sympy_answer is None else sympy_answer

    def solve(self, question: str) -> SolveResult:  # noqa: ARG002
        return SolveResult(
            final_answer=self.final_answer,
            sympy_answer=self.sympy_answer,
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
                        "sympy_answer": "4",
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
                controller=_FakeController("2 + 2", "2 + 2"),
                config=config,
            )

            metrics = runner.run()

            self.assertEqual(metrics["rows"], 1)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertTrue((tmp_path / "predictions.jsonl").exists())
            self.assertTrue((tmp_path / "tool_traces.jsonl").exists())
            self.assertTrue((tmp_path / "metrics.json").exists())
            prediction_row = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(prediction_row["predicted_answer"], "2 + 2")
            self.assertEqual(prediction_row["predicted_sympy_answer"], "2 + 2")
            self.assertEqual(prediction_row["reference_sympy_answer"], "4")
            self.assertEqual(prediction_row["match_type"], "symbolic")

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
                                "sympy_answer": "4",
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
                controller=_FakeController("4", "4"),
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
                        "sympy_answer": "4",
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

            runner = hu.instantiate(cfg, controller=_FakeController("4", "4"))
            metrics = runner.run()

            self.assertEqual(metrics["rows"], 1)
            self.assertEqual(metrics["accuracy"], 1.0)

    def test_compares_list_sympy_answers_elementwise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "problem-1",
                        "question": "Solve the system.",
                        "answer": "x = n + 1, y = n - 1",
                        "sympy_answer": ["n + 1", "n - 1"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            runner = BenchmarkRunner(
                controller=_FakeController("x = n + 1, y = n - 1", ["1 + n", "n - 1"]),
                config=BenchmarkConfig(dataset_path=dataset_path, output_dir=tmp_path, limit=1),
            )

            metrics = runner.run()

            self.assertEqual(metrics["accuracy"], 1.0)
            prediction_row = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(prediction_row["match_type"], "symbolic")

    def test_compares_single_tuple_sympy_answers_as_one_object(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            dataset_path = tmp_path / "data.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "id": "problem-1",
                        "question": "Find the point.",
                        "answer": "(n + 1, 2n)",
                        "sympy_answer": "Tuple(n + 1, 2*n)",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            runner = BenchmarkRunner(
                controller=_FakeController("(n + 1, 2n)", "Tuple(1 + n, n + n)"),
                config=BenchmarkConfig(dataset_path=dataset_path, output_dir=tmp_path, limit=1),
            )

            metrics = runner.run()

            self.assertEqual(metrics["accuracy"], 1.0)
            prediction_row = json.loads((tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip())
            self.assertEqual(prediction_row["match_type"], "symbolic")


if __name__ == "__main__":
    unittest.main()
