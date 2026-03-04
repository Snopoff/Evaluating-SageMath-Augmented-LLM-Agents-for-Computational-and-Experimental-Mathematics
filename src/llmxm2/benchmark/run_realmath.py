from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.llmxm2.agent.controller import AgentController
from src.llmxm2.mcp.client import SageToolClient

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

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], dataset_path: Path) -> "BenchmarkConfig":
        cfg = dict(cfg)
        output_dir = Path(str(cfg.get("output_dir", ".")))
        return cls(
            dataset_path=dataset_path,
            output_dir=output_dir,
            limit=int(cfg.get("limit", 25)),
            progress_logs=bool(cfg.get("progress_logs", False)),
            question_field=str(cfg.get("question_field", "question")),
            answer_field=str(cfg.get("answer_field", "answer")),
            id_field=str(cfg.get("id_field", "id")),
            predictions_file=str(cfg.get("predictions_file", "predictions.jsonl")),
            tool_traces_file=str(cfg.get("tool_traces_file", "tool_traces.jsonl")),
            metrics_file=str(cfg.get("metrics_file", "metrics.json")),
        )


@dataclass(frozen=True)
class ScoreResult:
    correct: bool
    match_type: str
    normalized_prediction: str
    normalized_reference: str


class RealMathBenchmarkRunner:
    def __init__(self, controller: AgentController, tool_client: SageToolClient, config: BenchmarkConfig):
        self.controller = controller
        self.tool_client = tool_client
        self.config = config

    def run(self) -> dict[str, Any]:
        self._progress("loading dataset rows")
        rows = list(self._iter_rows(self.config.dataset_path, self.config.limit))
        self._progress(f"loaded rows: {len(rows)}")
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

                self._progress(f"solving problem {total + 1}/{len(rows)} id={problem_id}")
                solve_result = self.controller.solve(question)
                score = self._score_prediction(solve_result.final_answer, reference_answer)

                total += 1
                if score.correct:
                    correct += 1
                    if score.match_type == "exact":
                        exact_correct += 1
                    elif score.match_type == "symbolic":
                        symbolic_correct += 1

                prediction_record = {
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
                }
                pred_handle.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")

                trace_record = {
                    "id": problem_id,
                    "tool_traces": solve_result.tool_traces,
                }
                trace_handle.write(json.dumps(trace_record, ensure_ascii=False) + "\n")
                self._progress(f"completed id={problem_id} (correct={score.correct}, tool_calls={len(solve_result.tool_traces)})")

        metrics = {
            "rows": total,
            "accuracy": round((correct / total), 6) if total else 0.0,
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

        self._progress(f"metrics written to {metrics_path}")
        return metrics

    def _progress(self, message: str) -> None:
        if self.config.progress_logs:
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

        if self._looks_symbolic(norm_prediction) and self._looks_symbolic(norm_reference):
            equivalent = self._symbolic_equivalent(norm_prediction, norm_reference)
            if equivalent:
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
        lhs_expr = self._equation_to_expression(lhs)
        rhs_expr = self._equation_to_expression(rhs)
        check_expr = f"(({lhs_expr})-({rhs_expr}))"
        check_expr_literal = json.dumps(check_expr)
        snippet = (
            f"expr = SR({check_expr_literal})\n"
            "RESULT = bool(expr.simplify_full() == 0)"
        )

        payload = {
            "operation": "sage_snippet",
            "args": {"code": snippet, "result_var": "RESULT"},
            "assumptions": {"domain": "QQ", "symbols": []},
            "request_id": str(uuid.uuid4()),
            "budget_profile": "conservative",
        }
        response = self.tool_client.sage_eval(payload)
        if response.get("status") != "ok":
            return False

        plain = str(response.get("result_plain", ""))
        result_repr = ""
        try:
            parsed = ast.literal_eval(plain)
            if isinstance(parsed, dict):
                result_repr = str(parsed.get("result_repr", ""))
        except (ValueError, SyntaxError):
            result_repr = ""

        if result_repr:
            return normalize_answer(result_repr).lower() in {"true", "1"}
        return normalize_answer(plain).lower() in {"true", "1"}

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
    text = text.removeprefix("[").removesuffix("]") if text.startswith("[") and text.endswith("]") else text
    return text
