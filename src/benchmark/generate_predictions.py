import json
import os
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any

from src.agent.controller import AgentController
from src.utils.console_logging import ConsoleLogger


@dataclass(frozen=True)
class GeneratePredictionsConfig:
    """Configuration for prediction-only dataset runs."""

    dataset_path: Path
    output_dir: Path
    limit: int = 25
    progress_logs: bool = False
    separate_logger_runs: bool = True
    sleep_sec_between_problems: float = 2.0
    max_attempts_per_problem: int = 3
    retry_backoff_sec: float = 10.0
    retry_backoff_multiplier: float = 2.0
    continue_on_problem_error: bool = True
    fsync_each_row: bool = False
    question_field: str = "question"
    ground_truth_field: str = "answer"
    ground_truth_sympy_field: str = "sympy_answer"
    id_field: str = "id"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_path", Path(self.dataset_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))


class GeneratePredictionsRunner:
    """Runs the agent on a dataset without scoring answers."""

    FILE_TIMESTAMP_FORMAT = "%Y-%m-%d-%H-%M-%S-%f"

    def __init__(
        self,
        controller: AgentController,
        config: GeneratePredictionsConfig,
        logger: ConsoleLogger | None = None,
    ) -> None:
        self.controller = controller
        self.config = config
        self.logger = logger

    def run(self) -> dict[str, Any]:
        rows = self._load_rows(self.config.dataset_path, self.config.limit)
        self._progress(f"loaded rows={len(rows)}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        predictions_path, summary_path = self._allocate_output_paths()
        summary = {
            "model": self._resolve_model_name(),
            "rows": len(rows),
            "completed_rows": 0,
            "successful_rows": 0,
            "failed_rows": 0,
            "dataset_path": str(self.config.dataset_path),
            "predictions_file": str(predictions_path),
            "summary_file": str(summary_path),
        }
        self._write_summary(summary_path, summary)

        with predictions_path.open("x", encoding="utf-8") as handle:
            for index, row in enumerate(rows):
                problem_id = row.get(self.config.id_field)
                if not isinstance(problem_id, str):
                    problem_id = f"row-{index + 1:05d}"

                question = str(row.get(self.config.question_field, ""))
                ground_truth = str(row.get(self.config.ground_truth_field, ""))
                ground_truth_sympy_answer = row.get(self.config.ground_truth_sympy_field, "")

                try:
                    solve_result, solve_time_sec = self._solve_problem(
                        question=question,
                        problem_id=problem_id,
                    )
                    payload = {
                        "id": problem_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "ground_truth_sympy_answer": ground_truth_sympy_answer,
                        "model_final_answer": solve_result.final_answer,
                        "sympy_answer": solve_result.sympy_answer,
                        "model_sympy_answer": solve_result.sympy_answer,
                        "explanation": solve_result.explanation,
                        "confidence": solve_result.confidence,
                        "verified_claims": solve_result.verified_claims,
                        "tool_traces": solve_result.tool_traces,
                        "stop_reason": solve_result.stop_reason,
                        "turn_count": solve_result.turn_count,
                        "token_usage": solve_result.token_usage,
                        "solve_time_sec": round(solve_time_sec, 6),
                        "error": None,
                    }
                    summary["successful_rows"] = int(summary.get("successful_rows", 0)) + 1
                except Exception as exc:
                    if not self.config.continue_on_problem_error:
                        raise
                    payload = self._build_error_payload(
                        problem_id=problem_id,
                        question=question,
                        ground_truth=ground_truth,
                        ground_truth_sympy_answer=ground_truth_sympy_answer,
                        error=exc,
                    )
                    summary["failed_rows"] = int(summary.get("failed_rows", 0)) + 1
                    self._progress(f"problem_id={problem_id} failed after retries: {exc.__class__.__name__}: {exc}")

                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                if self.config.fsync_each_row:
                    os.fsync(handle.fileno())

                summary["completed_rows"] = index + 1
                summary["last_problem_id"] = problem_id
                self._write_summary(summary_path, summary)
                if index + 1 < len(rows) and self.config.sleep_sec_between_problems > 0:
                    time.sleep(self.config.sleep_sec_between_problems)

        if self.logger is not None and hasattr(self.logger, "log_artifact"):
            artifact_metadata = {
                "dataset_path": str(self.config.dataset_path),
                "limit": self.config.limit,
                "rows": len(rows),
            }
            self.logger.log_artifact(
                name="generate-predictions",
                path=predictions_path,
                artifact_type="generate-predictions",
                metadata=artifact_metadata,
            )
            self.logger.log_artifact(
                name="generate-predictions-summary",
                path=summary_path,
                artifact_type="generate-predictions-summary",
                metadata=artifact_metadata,
            )

        return summary

    def _allocate_output_paths(self) -> tuple[Path, Path]:
        for _ in range(1000):
            timestamp = datetime.now().strftime(self.FILE_TIMESTAMP_FORMAT)
            predictions_path = self.config.output_dir / f"predictions_{timestamp}.jsonl"
            summary_path = self.config.output_dir / f"summary_{timestamp}.json"
            if not predictions_path.exists() and not summary_path.exists():
                return predictions_path, summary_path
            time.sleep(0.001)
        raise RuntimeError("Unable to allocate unique prediction output paths.")

    def _resolve_model_name(self) -> str:
        model_name = getattr(self.controller, "model_name", "")
        if isinstance(model_name, str) and model_name.strip():
            return model_name
        wrapped_model = getattr(self.controller, "model", None)
        fallback = getattr(wrapped_model, "model_name", "")
        return fallback if isinstance(fallback, str) else ""

    def _progress(self, message: str) -> None:
        if self.config.progress_logs:
            if self.logger is not None:
                self.logger.progress(f"[generate_predictions] {message}")

    def _finish_problem_run(self, logger: ConsoleLogger | None, *, status: str) -> None:
        if logger is None or not self.config.separate_logger_runs:
            return
        if logger.run_active:
            logger.finish_run(status=status)

    def _run_logger(self) -> ConsoleLogger | None:
        controller_logger = getattr(self.controller, "logger", None)
        if isinstance(controller_logger, ConsoleLogger):
            return controller_logger
        return self.logger

    def _solve_problem(
        self,
        *,
        question: str,
        problem_id: str,
    ) -> tuple[Any, float]:
        run_logger = self._run_logger()
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_attempts_per_problem + 1):
            self._start_problem_run(
                run_logger,
                question=question,
                problem_id=problem_id,
                attempt=attempt,
            )
            solve_started_at = time.perf_counter()
            try:
                solve_result = self.controller.solve(question)
            except Exception as exc:
                self._finish_problem_run(run_logger, status="failed")
                if not self._is_retryable_problem_error(exc) or attempt >= self.config.max_attempts_per_problem:
                    raise
                last_error = exc
                sleep_sec = self.config.retry_backoff_sec * (self.config.retry_backoff_multiplier ** (attempt - 1))
                self._progress(
                    f"retrying problem_id={problem_id} attempt={attempt + 1}/{self.config.max_attempts_per_problem} "
                    f"after {exc.__class__.__name__}: sleeping {sleep_sec:.1f}s"
                )
                time.sleep(sleep_sec)
                continue
            solve_time_sec = time.perf_counter() - solve_started_at
            self._finish_problem_run(run_logger, status=solve_result.stop_reason)
            return solve_result, solve_time_sec

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to solve problem {problem_id!r}.")

    def _start_problem_run(
        self,
        logger: ConsoleLogger | None,
        *,
        question: str,
        problem_id: str,
        attempt: int,
    ) -> None:
        if logger is None or not self.config.separate_logger_runs:
            return
        logger.start_run(
            metadata={
                "problem_id": problem_id,
                "question": question,
                "problem_attempt": attempt,
            }
        )

    @staticmethod
    def _build_error_payload(
        *,
        problem_id: str,
        question: str,
        ground_truth: str,
        ground_truth_sympy_answer: str | list[str],
        error: Exception,
    ) -> dict[str, Any]:
        return {
            "id": problem_id,
            "question": question,
            "ground_truth": ground_truth,
            "ground_truth_sympy_answer": ground_truth_sympy_answer,
            "model_final_answer": "",
            "sympy_answer": "",
            "model_sympy_answer": "",
            "explanation": "",
            "confidence": 0,
            "verified_claims": [],
            "tool_traces": [],
            "stop_reason": "problem_failed",
            "turn_count": 0,
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "solve_time_sec": 0.0,
            "error": {
                "type": error.__class__.__name__,
                "message": str(error),
            },
        }

    @staticmethod
    def _is_retryable_problem_error(exc: Exception) -> bool:
        retryable_class_names = {
            "APITimeoutError",
            "APIConnectionError",
            "RateLimitError",
            "TimeoutError",
            "ReadTimeout",
            "ConnectTimeout",
            "RemoteProtocolError",
        }
        current: BaseException | None = exc
        while current is not None:
            if current.__class__.__name__ in retryable_class_names:
                return True
            message = str(current).lower()
            if any(marker in message for marker in ("timed out", "timeout", "rate limit", "too many requests", "429")):
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _write_summary(path: Path, summary: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)

    @staticmethod
    def _load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("problems"), list):
            rows = [row for row in payload["problems"] if isinstance(row, dict)]
        elif isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            rows = []

        if limit == -1:
            return rows
        return rows[:limit]
