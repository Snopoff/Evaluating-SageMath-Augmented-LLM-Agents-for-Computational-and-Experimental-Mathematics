from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import hydra.utils as hu

from src.benchmark.judge_sympy_answers import DEFAULT_MAX_TOKENS


DEFAULT_TAGS_PATH = Path("data/results/arxiv_tags_by_problem.json")
DEFAULT_ANSWER_TYPES_PATH = Path("data/processed/normalized_problems.json")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmarking already-generated prediction rows."""

    predictions_path: Path
    tags_path: Path = DEFAULT_TAGS_PATH
    answer_types_path: Path | None = DEFAULT_ANSWER_TYPES_PATH
    output_dir: Path | None = None
    limit: int = -1
    per_row_timeout_sec: float = 2.0
    id_field: str = "id"
    question_field: str = "question"
    correct_field: str = "correct"
    prediction_sympy_fields: tuple[str, ...] = ("model_sympy_answer", "predicted_sympy_answer", "sympy_answer")
    reference_sympy_fields: tuple[str, ...] = ("ground_truth_sympy_answer", "reference_sympy_answer", "sympy_answer")
    judge_specs: tuple[str, ...] = ()
    judge_limit: int = 0
    judge_resume: bool = False
    judge_max_tokens: int = DEFAULT_MAX_TOKENS
    wilson_confidence_level: float = 0.95
    progress_logs: bool = True
    print_statistics: bool = True
    strict_metadata: bool = False
    sympy_output_filename: str | None = None
    sympy_summary_filename: str = "sympy_summary.json"
    judge_output_filename: str = "judge_verdicts.json"
    final_output_filename: str | None = None
    statistics_filename: str = "statistics.json"

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictions_path", hydra_path(self.predictions_path))
        object.__setattr__(self, "tags_path", hydra_path(self.tags_path))
        if self.answer_types_path is not None:
            object.__setattr__(self, "answer_types_path", hydra_path(self.answer_types_path))
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", hydra_path(self.output_dir))
        object.__setattr__(self, "prediction_sympy_fields", tuple(self.prediction_sympy_fields))
        object.__setattr__(self, "reference_sympy_fields", tuple(self.reference_sympy_fields))
        object.__setattr__(self, "judge_specs", tuple(self.judge_specs))


def hydra_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(hu.to_absolute_path(str(path)))
