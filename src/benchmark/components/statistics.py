from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from src.benchmark.components.config import BenchmarkConfig
from src.benchmark.wilson_ci import wilson_summary


def build_statistics(
    *,
    config: BenchmarkConfig,
    output_dir: Path,
    sympy_output_path: Path,
    sympy_summary_path: Path,
    judge_output_path: Path,
    judge_summary_path: Path,
    final_output_path: Path,
    statistics_path: Path,
    sympy_summary: dict[str, Any],
    judge_summary: dict[str, Any],
    enrichment_stats: dict[str, Any],
    final_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    final_correct = sum(1 for row in final_rows if row.get("final_vote") is True)
    final_false = len(final_rows) - final_correct
    row_count = len(final_rows)
    sympy_correct = int(sympy_summary["correct"])
    sympy_rows = int(sympy_summary["rows"])
    sympy_wilson = _wilson_payload(
        correct=sympy_correct,
        rows=sympy_rows,
        confidence_level=config.wilson_confidence_level,
    )
    final_wilson = _wilson_payload(
        correct=final_correct,
        rows=row_count,
        confidence_level=config.wilson_confidence_level,
    )
    return {
        "input_path": str(config.predictions_path),
        "tags_path": str(config.tags_path),
        "answer_types_path": str(config.answer_types_path) if config.answer_types_path is not None else None,
        "output_dir": str(output_dir),
        "sympy_output_path": str(sympy_output_path),
        "sympy_summary_path": str(sympy_summary_path),
        "judge_output_path": str(judge_output_path),
        "judge_summary_path": str(judge_summary_path),
        "final_output_path": str(final_output_path),
        "statistics_path": str(statistics_path),
        "rows": row_count,
        "sympy_accuracy": sympy_summary["accuracy"],
        "sympy_correct": sympy_correct,
        "sympy_incorrect": sympy_summary["incorrect"],
        "sympy_wilson": sympy_wilson,
        "final_accuracy": round(final_correct / row_count, 6) if final_rows else 0.0,
        "final_correct": final_correct,
        "final_incorrect": final_false,
        "final_wilson": final_wilson,
        "wilson_confidence_level": config.wilson_confidence_level,
        "sympy_summary": sympy_summary,
        "judge_summary": judge_summary,
        **enrichment_stats,
    }


def _wilson_payload(*, correct: int, rows: int, confidence_level: float) -> dict[str, Any] | None:
    if rows <= 0:
        return None
    return wilson_summary(correct=correct, rows=rows, confidence_level=confidence_level).as_dict()
