from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from src.benchmark.compare_predictions import (
    _is_missing_sympy_answer,
    _resolve_first_present,
    _score_with_timeout,
)
from src.benchmark.components.config import BenchmarkConfig
from src.benchmark.components.io import write_json
from src.benchmark.sympy_compare import SympyAnswerComparator


def run_sympy_comparison(
    *,
    config: BenchmarkConfig,
    rows: Sequence[dict[str, Any]],
    malformed_rows_skipped: int,
    output_path: Path,
    summary_path: Path,
    comparator: SympyAnswerComparator | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    comparator = comparator or SympyAnswerComparator()
    total = 0
    correct = 0
    exact_correct = 0
    symbolic_correct = 0
    missing_prediction = 0
    missing_reference = 0
    missing_answer_type = 0
    timed_out_rows = 0
    results: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        problem_id = row.get(config.id_field)
        if not isinstance(problem_id, str) or not problem_id:
            problem_id = f"row-{index + 1:05d}"

        question = str(row.get(config.question_field, ""))
        prediction_value, prediction_field = _resolve_first_present(row, config.prediction_sympy_fields)
        reference_value, reference_field = _resolve_first_present(row, config.reference_sympy_fields)
        prediction_sympy = comparator.coerce(prediction_value)
        reference_sympy = comparator.coerce(reference_value)
        score, timed_out = _score_with_timeout(
            comparator=comparator,
            prediction=prediction_sympy,
            reference=reference_sympy,
            timeout_sec=config.per_row_timeout_sec,
        )

        total += 1
        if _is_missing_sympy_answer(prediction_sympy):
            missing_prediction += 1
        if _is_missing_sympy_answer(reference_sympy):
            missing_reference += 1
        if not row.get("answer_type"):
            missing_answer_type += 1
        if timed_out:
            timed_out_rows += 1
        if score.correct:
            correct += 1
            if score.match_type == "exact":
                exact_correct += 1
            if score.match_type == "symbolic":
                symbolic_correct += 1

        result_row = dict(row)
        result_row.update(
            {
                "id": problem_id,
                "question": question,
                "prediction_sympy_answer": prediction_sympy,
                "reference_sympy_answer": reference_sympy,
                "prediction_sympy_field": prediction_field,
                "reference_sympy_field": reference_field,
                "normalized_prediction": score.normalized_prediction,
                "normalized_reference": score.normalized_reference,
                "correct": score.correct,
                "matches_reference": score.correct,
                "match_type": score.match_type,
            }
        )
        results.append(result_row)

    summary = {
        "rows": total,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "correct": correct,
        "incorrect": total - correct,
        "exact_correct": exact_correct,
        "symbolic_correct": symbolic_correct,
        "missing_prediction_rows": missing_prediction,
        "missing_reference_rows": missing_reference,
        "missing_answer_type_rows": missing_answer_type,
        "timed_out_rows": timed_out_rows,
        "malformed_rows_skipped": malformed_rows_skipped,
        "input_path": str(config.predictions_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "per_row_timeout_sec": config.per_row_timeout_sec,
        "prediction_sympy_fields": list(config.prediction_sympy_fields),
        "reference_sympy_fields": list(config.reference_sympy_fields),
    }
    write_json(output_path, results)
    write_json(summary_path, summary)
    if progress is not None:
        progress(f"sympy comparison wrote {output_path}")
    return summary
