import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.benchmark.compare_predictions import ComparePredictionsConfig, compare_predictions  # noqa: E402
from src.benchmark.sympy_compare import ScoreResult  # noqa: E402


class ComparePredictionsTests(unittest.TestCase):
    def test_compares_generate_predictions_rows_with_model_sympy_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "predictions.json"
            output_path = tmp_path / "comparison.jsonl"
            summary_path = tmp_path / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "problem-1",
                        "question": "What is 2+2?",
                        "ground_truth_sympy_answer": "4",
                        "model_sympy_answer": "2 + 2",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "id": "problem-2",
                        "question": "Solve the system.",
                        "ground_truth_sympy_answer": ["n + 1", "n - 1"],
                        "model_sympy_answer": ["1 + n", "n - 1"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = compare_predictions(
                ComparePredictionsConfig(
                    input_path=input_path,
                    output_path=output_path,
                    summary_path=summary_path,
                )
            )

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["accuracy"], 1.0)
            self.assertEqual(summary["symbolic_correct"], 2)
            self.assertEqual(summary["malformed_rows_skipped"], 0)
            self.assertEqual(summary["timed_out_rows"], 0)

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(rows[0]["prediction_sympy_field"], "model_sympy_answer")
            self.assertEqual(rows[0]["reference_sympy_field"], "ground_truth_sympy_answer")
            self.assertEqual(rows[0]["match_type"], "symbolic")
            self.assertTrue(rows[0]["correct"])
            self.assertEqual(rows[1]["match_type"], "symbolic")
            self.assertTrue(rows[1]["correct"])

    def test_supports_benchmark_field_names_and_tracks_missing_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "benchmark_predictions.jsonl"
            output_path = tmp_path / "comparison.jsonl"
            summary_path = tmp_path / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "problem-1",
                        "question": "Find the point.",
                        "reference_sympy_answer": "Tuple(n + 1, 2*n)",
                        "predicted_sympy_answer": "Tuple(1 + n, n + n)",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "id": "problem-2",
                        "question": "What is 2+2?",
                        "reference_sympy_answer": "4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = compare_predictions(
                ComparePredictionsConfig(
                    input_path=input_path,
                    output_path=output_path,
                    summary_path=summary_path,
                )
            )

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["correct"], 1)
            self.assertEqual(summary["missing_prediction_rows"], 1)
            self.assertEqual(summary["accuracy"], 0.5)
            self.assertEqual(summary["malformed_rows_skipped"], 0)
            self.assertEqual(summary["timed_out_rows"], 0)

            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(rows[0]["prediction_sympy_field"], "predicted_sympy_answer")
            self.assertEqual(rows[0]["reference_sympy_field"], "reference_sympy_answer")
            self.assertEqual(rows[0]["match_type"], "symbolic")
            self.assertEqual(rows[1]["prediction_sympy_answer"], "")
            self.assertEqual(rows[1]["match_type"], "mismatch")

    def test_salvages_valid_rows_from_malformed_object_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "predictions.json"
            output_path = tmp_path / "comparison.jsonl"
            summary_path = tmp_path / "summary.json"
            input_path.write_text(
                '{\n'
                '  "id": "problem-1",\n'
                '  "ground_truth_sympy_answer": "4",\n'
                '  "model_sympy_answer": "2 + 2",\n'
                '  "solve_time_sec": 1.0,\n'
                '  {\n'
                '    "id": "problem-2",\n'
                '    "ground_truth_sympy_answer": "Tuple(n + 1, 2*n)",\n'
                '    "model_sympy_answer": "Tuple(1 + n, n + n)",\n'
                '    "error": null\n'
                '  }\n'
                '{\n'
                '  "id": "problem-3",\n'
                '  "ground_truth_sympy_answer": "0",\n'
                '  "model_sympy_answer": "0",\n'
                '  "error": null\n'
                '}\n',
                encoding="utf-8",
            )

            summary = compare_predictions(
                ComparePredictionsConfig(
                    input_path=input_path,
                    output_path=output_path,
                    summary_path=summary_path,
                )
            )

            self.assertEqual(summary["rows"], 2)
            self.assertEqual(summary["correct"], 2)
            self.assertEqual(summary["malformed_rows_skipped"], 1)
            self.assertEqual(summary["timed_out_rows"], 0)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual([row["id"] for row in rows], ["problem-2", "problem-3"])

    def test_marks_timed_out_rows_without_hanging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "predictions.jsonl"
            output_path = tmp_path / "comparison.jsonl"
            summary_path = tmp_path / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "problem-1",
                        "ground_truth_sympy_answer": "4",
                        "model_sympy_answer": "2 + 2",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("src.benchmark.compare_predictions._score_with_timeout") as mocked_score:
                mocked_score.return_value = (
                    ScoreResult(
                        correct=False,
                        match_type="timeout",
                        normalized_prediction="2 + 2",
                        normalized_reference="4",
                    ),
                    True,
                )
                summary = compare_predictions(
                    ComparePredictionsConfig(
                        input_path=input_path,
                        output_path=output_path,
                        summary_path=summary_path,
                    )
                )

            self.assertEqual(summary["rows"], 1)
            self.assertEqual(summary["timed_out_rows"], 1)
            self.assertEqual(summary["accuracy"], 0.0)
            row = json.loads(output_path.read_text(encoding="utf-8").strip())
            self.assertEqual(row["match_type"], "timeout")
            self.assertFalse(row["correct"])


if __name__ == "__main__":
    unittest.main()
