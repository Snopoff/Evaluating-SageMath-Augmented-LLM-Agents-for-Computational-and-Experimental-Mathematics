from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llmxm2.agent.controller import SolveResult
from llmxm2.benchmark.run_realmath import BenchmarkConfig, RealMathBenchmarkRunner


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
            runner = RealMathBenchmarkRunner(
                controller=_FakeController("4"),
                config=config,
            )

            metrics = runner.run()

            self.assertEqual(metrics["rows"], 1)
            self.assertEqual(metrics["accuracy"], 1.0)
            self.assertTrue((tmp_path / "predictions.jsonl").exists())
            self.assertTrue((tmp_path / "tool_traces.jsonl").exists())
            self.assertTrue((tmp_path / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
