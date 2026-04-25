from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.agent.controller import AgentController
from src.sage.runtime import SageRuntime

_LATEX_WRAPPERS = [
    (r"\\[", ""),
    (r"\\]", ""),
    (r"\\(", ""),
    (r"\\)", ""),
    ("$", ""),
    (r"\\left", ""),
    (r"\\right", ""),
]
_ALLOWED_SYMBOLIC_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/^=()\[\]{}.,<>| :]+$")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for running the RealMath benchmark loop.

    Args:
        dataset_path: Input dataset path containing benchmark rows.
        output_dir: Directory where predictions, traces, and metrics are written.
        limit: Maximum number of dataset rows to process.
        progress_logs: Whether benchmark progress messages are enabled.
        question_field: Input field name containing the benchmark prompt.
        answer_field: Input field name containing the reference answer.
        id_field: Input field name used as the row identifier.
        predictions_file: Output filename for prediction records.
        tool_traces_file: Output filename for tool trace records.
        metrics_file: Output filename for aggregate metrics.
        symbolic_timeout_sec: Timeout used for symbolic Sage equivalence checks.
    """

    dataset_path: Path
    output_dir: Path
    limit: int = 25
    progress_logs: bool = False
    question_field: str = "question"
    answer_field: str = "answer"
    id_field: str = "id"
    predictions_file: str = "predictions.jsonl"
    tool_traces_file: str = "tool_traces.jsonl"
    metrics_file: str = "metrics.json"
    symbolic_timeout_sec: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], dataset_path: Path) -> BenchmarkConfig:
        cfg_dict = dict(cfg)
        output_dir = Path(str(cfg_dict.get("output_dir", ".")))
        return cls(
            dataset_path=dataset_path,
            output_dir=output_dir,
            limit=int(cfg_dict.get("limit", 25)),
            progress_logs=bool(cfg_dict.get("progress_logs", False)),
            question_field=str(cfg_dict.get("question_field", "question")),
            answer_field=str(cfg_dict.get("answer_field", "answer")),
            id_field=str(cfg_dict.get("id_field", "id")),
            predictions_file=str(cfg_dict.get("predictions_file", "predictions.jsonl")),
            tool_traces_file=str(cfg_dict.get("tool_traces_file", "tool_traces.jsonl")),
            metrics_file=str(cfg_dict.get("metrics_file", "metrics.json")),
            symbolic_timeout_sec=float(cfg_dict.get("symbolic_timeout_sec", 8.0)),
        )


@dataclass(frozen=True)
class ScoreResult:
    """Normalized comparison result for one prediction/reference pair.

    Args:
        correct: Whether the prediction matched the reference.
        match_type: Match mode such as ``exact`` or ``symbolic``.
        normalized_prediction: Canonicalized predicted answer.
        normalized_reference: Canonicalized reference answer.
    """

    correct: bool
    match_type: str
    normalized_prediction: str
    normalized_reference: str


class RealMathBenchmarkRunner:
    """Executes benchmark rows and writes predictions, traces, and metrics.

    Args:
        controller: Controller used to answer benchmark questions.
        config: Benchmark configuration and output locations.
        sage_runtime: Optional Sage runtime used for symbolic equivalence checks.
    """

    def __init__(
        self,
        controller: AgentController,
        config: BenchmarkConfig,
        sage_runtime: SageRuntime | None = None,
        logger: Any | None = None,
    ):
        self.controller = controller
        self.config = config
        self.sage_runtime = sage_runtime
        self.logger = logger

    def run(self) -> dict[str, Any]:
        rows = list(self._iter_rows(self.config.dataset_path, self.config.limit))
        self._progress(f"loaded rows={len(rows)}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        predictions_path = self.config.output_dir / self.config.predictions_file
        traces_path = self.config.output_dir / self.config.tool_traces_file
        metrics_path = self.config.output_dir / self.config.metrics_file

        total = 0
        correct = 0
        exact_correct = 0
        symbolic_correct = 0

        with predictions_path.open("w", encoding="utf-8") as pred_handle, traces_path.open("w", encoding="utf-8") as trace_handle:
            for row in rows:
                problem_id = row.get(self.config.id_field)
                if not isinstance(problem_id, str):
                    problem_id = f"row-{total + 1:05d}"

                question = str(row.get(self.config.question_field, ""))
                reference_answer = str(row.get(self.config.answer_field, ""))

                solve_result = self.controller.solve(question)
                score = self._score_prediction(solve_result.final_answer, reference_answer)

                total += 1
                if score.correct:
                    correct += 1
                    if score.match_type == "exact":
                        exact_correct += 1
                    if score.match_type == "symbolic":
                        symbolic_correct += 1

                pred_handle.write(
                    json.dumps(
                        {
                            "id": problem_id,
                            "question": question,
                            "reference_answer": reference_answer,
                            "predicted_answer": solve_result.final_answer,
                            "normalized_prediction": score.normalized_prediction,
                            "normalized_reference": score.normalized_reference,
                            "correct": score.correct,
                            "match_type": score.match_type,
                            "tool_calls": len(solve_result.tool_traces),
                            "turn_count": solve_result.turn_count,
                            "stop_reason": solve_result.stop_reason,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                trace_handle.write(
                    json.dumps(
                        {
                            "id": problem_id,
                            "tool_traces": solve_result.tool_traces,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        metrics = {
            "rows": total,
            "accuracy": round(correct / total, 6) if total else 0.0,
            "correct": correct,
            "incorrect": total - correct,
            "exact_correct": exact_correct,
            "symbolic_correct": symbolic_correct,
            "predictions_file": str(predictions_path),
            "tool_traces_file": str(traces_path),
            "metrics_file": str(metrics_path),
        }

        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, ensure_ascii=False)

        return metrics

    def _progress(self, message: str) -> None:
        if self.config.progress_logs:
            if self.logger is not None:
                self.logger.progress(f"[benchmark] {message}")
                return
            print(f"[progress][benchmark] {message}", flush=True)

    @staticmethod
    def _iter_rows(path: Path, limit: int) -> Iterable[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= limit:
                    break
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    yield payload

    def _score_prediction(self, prediction: str, reference: str) -> ScoreResult:
        norm_prediction = normalize_answer(prediction)
        norm_reference = normalize_answer(reference)

        if norm_prediction == norm_reference:
            return ScoreResult(
                correct=True,
                match_type="exact",
                normalized_prediction=norm_prediction,
                normalized_reference=norm_reference,
            )

        if self.sage_runtime and self._looks_symbolic(norm_prediction) and self._looks_symbolic(norm_reference):
            if self._symbolic_equivalent(norm_prediction, norm_reference):
                return ScoreResult(
                    correct=True,
                    match_type="symbolic",
                    normalized_prediction=norm_prediction,
                    normalized_reference=norm_reference,
                )

        return ScoreResult(
            correct=False,
            match_type="mismatch",
            normalized_prediction=norm_prediction,
            normalized_reference=norm_reference,
        )

    def _symbolic_equivalent(self, lhs: str, rhs: str) -> bool:
        if self.sage_runtime is None:
            return False

        lhs_expr = self._equation_to_expression(lhs)
        rhs_expr = self._equation_to_expression(rhs)
        snippet = f"expr = SR((({lhs_expr})-({rhs_expr})))\nRESULT = bool(expr.simplify_full() == 0)"

        result = self.sage_runtime.execute_sage_code(
            code=snippet,
            result_var="RESULT",
            timeout_sec=self.config.symbolic_timeout_sec,
        )
        if result.status != "ok":
            return False

        return normalize_answer(result.result_plain).lower() in {"true", "1"}

    @staticmethod
    def _looks_symbolic(value: str) -> bool:
        if not value or len(value) > 400:
            return False
        if "\\" in value:
            return False
        if re.search(r"[A-Za-z]{4,}", value):
            return False
        return bool(_ALLOWED_SYMBOLIC_CHARS.match(value))

    @staticmethod
    def _equation_to_expression(value: str) -> str:
        if "==" in value:
            left, right = value.split("==", 1)
            return f"(({left})-({right}))"
        if "=" in value:
            left, right = value.split("=", 1)
            return f"(({left})-({right}))"
        return value


def normalize_answer(value: str) -> str:
    text = value.strip()
    for old, new in _LATEX_WRAPPERS:
        text = text.replace(old, new)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text.removeprefix("[").removesuffix("]")
    return text
