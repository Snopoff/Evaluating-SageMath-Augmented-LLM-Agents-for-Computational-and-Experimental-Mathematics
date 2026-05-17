import json
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.benchmark.sympy_compare import ScoreResult, SympyAnswerComparator


@dataclass(frozen=True)
class ComparePredictionsConfig:
    input_path: Path
    output_path: Path | None = None
    summary_path: Path | None = None
    limit: int = -1
    id_field: str = "id"
    question_field: str = "question"
    per_row_timeout_sec: float = 2.0
    prediction_sympy_fields: tuple[str, ...] = ("model_sympy_answer", "predicted_sympy_answer", "sympy_answer")
    reference_sympy_fields: tuple[str, ...] = ("ground_truth_sympy_answer", "reference_sympy_answer", "sympy_answer")

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        object.__setattr__(self, "output_path", Path(self.output_path) if self.output_path is not None else None)
        object.__setattr__(self, "summary_path", Path(self.summary_path) if self.summary_path is not None else None)


def compare_predictions(config: ComparePredictionsConfig) -> dict[str, Any]:
    rows, malformed_rows_skipped = _load_rows(config.input_path, config.limit)
    comparator = SympyAnswerComparator()
    output_path = config.output_path or _default_output_path(config.input_path)
    summary_path = config.summary_path or _default_summary_path(config.input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    exact_correct = 0
    symbolic_correct = 0
    missing_prediction = 0
    missing_reference = 0
    timed_out_rows = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            problem_id = row.get(config.id_field)
            if not isinstance(problem_id, str):
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
            if timed_out:
                timed_out_rows += 1
            if score.correct:
                correct += 1
                if score.match_type == "exact":
                    exact_correct += 1
                if score.match_type == "symbolic":
                    print(f"Symbolic match for problem ID {problem_id}")
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
                    "match_type": score.match_type,
                }
            )
            handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")

    summary = {
        "rows": total,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "correct": correct,
        "incorrect": total - correct,
        "exact_correct": exact_correct,
        "symbolic_correct": symbolic_correct,
        "missing_prediction_rows": missing_prediction,
        "missing_reference_rows": missing_reference,
        "timed_out_rows": timed_out_rows,
        "malformed_rows_skipped": malformed_rows_skipped,
        "input_path": str(config.input_path),
        "output_path": str(output_path),
        "summary_path": str(summary_path),
        "per_row_timeout_sec": config.per_row_timeout_sec,
        "prediction_sympy_fields": list(config.prediction_sympy_fields),
        "reference_sympy_fields": list(config.reference_sympy_fields),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return summary


def _load_rows(path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    if path.suffix == ".json":
        return _load_json_rows(path, limit)
    return _load_jsonl_rows(path, limit)


def _load_jsonl_rows(path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed_rows_skipped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows_skipped += 1
                continue
            if isinstance(payload, dict):
                rows.append(payload)
                if limit != -1 and len(rows) >= limit:
                    break
    return rows, malformed_rows_skipped


def _load_json_rows(path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _load_object_stream_rows(text, limit)

    if isinstance(payload, dict) and isinstance(payload.get("problems"), list):
        rows = [row for row in payload["problems"] if isinstance(row, dict)]
    elif isinstance(payload, dict) and isinstance(payload.get("results"), list):
        rows = [row for row in payload["results"] if isinstance(row, dict)]
    elif isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        rows = []

    if limit != -1:
        rows = rows[:limit]
    return rows, 0


def _load_object_stream_rows(text: str, limit: int) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed_rows_skipped = 0
    starts = _record_start_offsets(text)
    if not starts:
        return rows, 1

    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        try:
            payload = json.loads(chunk)
        except json.JSONDecodeError:
            malformed_rows_skipped += 1
            continue
        if isinstance(payload, dict):
            rows.append(payload)
            if limit != -1 and len(rows) >= limit:
                break

    return rows, malformed_rows_skipped


def _iter_rows(path: Path, limit: int) -> Iterable[dict[str, Any]]:
    if path.suffix == ".json":
        rows, _ = _load_json_rows(path, limit)
        yield from rows
        return

    rows, _ = _load_jsonl_rows(path, limit)
    yield from rows


def _iter_json_rows(path: Path, limit: int) -> Iterable[dict[str, Any]]:
    rows, _ = _load_json_rows(path, limit)
    yield from rows


def _iter_json_stream(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        payload, next_index = decoder.raw_decode(text, index)
        if isinstance(payload, dict):
            yield payload
        index = next_index


def _record_start_offsets(text: str) -> list[int]:
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    offset = 0
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if re.match(r'^\{"id"\s*:', stripped):
            offsets.append(offset)
        elif stripped.strip() == "{":
            next_index = index + 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            if next_index < len(lines) and lines[next_index].lstrip().startswith('"id"'):
                offsets.append(offset)
        offset += len(line)
    return offsets


def _resolve_first_present(row: dict[str, Any], field_names: Sequence[str]) -> tuple[Any, str]:
    for field_name in field_names:
        if field_name in row:
            return row[field_name], field_name
    return "", ""


def _is_missing_sympy_answer(value: str | list[str]) -> bool:
    if isinstance(value, list):
        return len(value) == 0 or all(not item.strip() for item in value)
    return not value.strip()


def _default_output_path(input_path: Path) -> Path:
    timestamp = time.strftime("%Y-%m-%d-%H-%M")
    return input_path.parent / f"{input_path.stem}_comparison_{timestamp}.jsonl"


def _default_summary_path(input_path: Path) -> Path:
    timestamp = time.strftime("%Y-%m-%d-%H-%M")
    return input_path.parent / f"{input_path.stem}_comparison_summary_{timestamp}.json"


class _RowTimeout(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: ARG001
    raise _RowTimeout()


def _score_with_timeout(
    *,
    comparator: SympyAnswerComparator,
    prediction: str | list[str],
    reference: str | list[str],
    timeout_sec: float,
) -> tuple[ScoreResult, bool]:
    if timeout_sec <= 0:
        return comparator.score(prediction, reference), False

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        return comparator.score(prediction, reference), False
    except _RowTimeout:
        normalized_prediction = comparator.normalize(prediction)
        normalized_reference = comparator.normalize(reference)
        return (
            ScoreResult(
                correct=True,
                match_type="exact",
                normalized_prediction=normalized_prediction,
                normalized_reference=normalized_reference,
            )
            if normalized_prediction == normalized_reference
            else ScoreResult(
                correct=False,
                match_type="timeout",
                normalized_prediction=normalized_prediction,
                normalized_reference=normalized_reference,
            ),
            True,
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
