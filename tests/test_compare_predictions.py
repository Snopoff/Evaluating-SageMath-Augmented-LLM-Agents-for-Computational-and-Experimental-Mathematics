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
            output_path = tmp_path / "comparison.json"
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
            self.assertEqual(summary["output_path"], str(output_path))
            self.assertEqual(summary["summary_path"], str(summary_path))
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), summary)
            rows = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["prediction_sympy_field"], "model_sympy_answer")
            self.assertEqual(rows[0]["reference_sympy_field"], "ground_truth_sympy_answer")
            self.assertEqual(rows[0]["match_type"], "symbolic")
            self.assertTrue(rows[0]["correct"])
            self.assertTrue(rows[0]["matches_reference"])
            self.assertEqual(rows[1]["match_type"], "symbolic")
            self.assertTrue(rows[1]["correct"])

    def test_supports_benchmark_field_names_and_tracks_missing_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "benchmark_predictions.jsonl"
            output_path = tmp_path / "comparison.json"
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
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), summary)
            rows = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(rows[0]["prediction_sympy_field"], "predicted_sympy_answer")
            self.assertEqual(rows[0]["reference_sympy_field"], "reference_sympy_answer")
            self.assertEqual(rows[0]["match_type"], "symbolic")
            self.assertEqual(rows[1]["prediction_sympy_answer"], "")
            self.assertEqual(rows[1]["match_type"], "mismatch")
            self.assertFalse(rows[1]["matches_reference"])

    def test_salvages_valid_rows_from_malformed_object_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "predictions.json"
            output_path = tmp_path / "comparison.json"
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
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), summary)
            rows = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([row["id"] for row in rows], ["problem-2", "problem-3"])

    def test_marks_timed_out_rows_without_hanging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "predictions.jsonl"
            output_path = tmp_path / "comparison.json"
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
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), summary)
            row = json.loads(output_path.read_text(encoding="utf-8"))[0]
            self.assertEqual(row["match_type"], "timeout")
            self.assertFalse(row["correct"])

    def test_allocates_default_summary_path_when_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "predictions.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "id": "problem-1",
                        "ground_truth_sympy_answer": "4",
                        "model_sympy_answer": "4",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = compare_predictions(ComparePredictionsConfig(input_path=input_path))

            output_path = Path(summary["output_path"])
            summary_path = Path(summary["summary_path"])
            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.parent, input_path.parent)
            self.assertRegex(output_path.name, r"^predictions_comparison_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.json$")
            self.assertTrue(summary_path.exists())
            self.assertEqual(summary_path.parent, input_path.parent)
            self.assertRegex(summary_path.name, r"^predictions_comparison_summary_\d{4}-\d{2}-\d{2}-\d{2}-\d{2}\.json$")
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), summary)

    def test_redirects_output_and_summary_when_requested_paths_match_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "predictions.json"
            original_rows = [
                {
                    "id": "problem-1",
                    "ground_truth_sympy_answer": "4",
                    "model_sympy_answer": "4",
                }
            ]
            input_path.write_text(json.dumps(original_rows, indent=2), encoding="utf-8")

            summary = compare_predictions(
                ComparePredictionsConfig(
                    input_path=input_path,
                    output_path=input_path,
                    summary_path=input_path,
                )
            )

            output_path = Path(summary["output_path"])
            summary_path = Path(summary["summary_path"])
            self.assertNotEqual(output_path, input_path)
            self.assertNotEqual(summary_path, input_path)
            self.assertEqual(output_path.suffix, ".json")
            self.assertEqual(summary_path, input_path.parent / "comparison" / input_path.name)
            self.assertEqual(json.loads(input_path.read_text(encoding="utf-8")), original_rows)
            self.assertEqual(len(json.loads(output_path.read_text(encoding="utf-8"))), 1)
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8")), summary)


if __name__ == "__main__":
    unittest.main()
