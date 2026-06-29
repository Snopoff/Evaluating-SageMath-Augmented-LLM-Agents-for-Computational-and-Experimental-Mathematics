from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import hydra.utils as hu
import rootutils
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

from src.utils.langchain_structured_output import structured_output_kwargs

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
load_dotenv()


DEFAULT_LIMIT = 0  # 0 = no limit after filtering to correct=false rows.
DEFAULT_INPUT_PATH = Path("data/results/deepseek/tool/output/sage_deepseekv32_medium_number_0_60.json")
DEFAULT_MAX_TOKENS = 4096
DEFAULT_MODEL_CONFIG_DIR = Path("configs/model")
DEFAULT_ENV_JUDGE_SPECS = "JUDGE_SPECS"

CONSOLE_FIELD_LIMIT = 250
JUDGE_COUNT = 3
LEGACY_FLAT_JUDGE_FIELD_RE = re.compile(r"^judge_\d+_.+")


class JudgeVerdict(BaseModel):
    """Structured answer expected from each LLM judge."""

    explanation: str = Field(description="Exactly one short sentence explaining the verdict.")
    verdict: Literal["yes", "no"] = Field(
        description=(
            "yes = the predicted SymPy answer is conceptually equivalent to the reference; no = the predicted answer genuinely differs."
        )
    )
    confidence: int = Field(ge=1, le=5, description="Confidence in the verdict on a 1 to 5 scale.")


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class JudgeSpec:
    """A model chosen from configs/model/*.yaml plus a concrete model name."""

    provider: str
    model_name: str
    label: str


@dataclass(frozen=True)
class JudgeConfig:
    input_path: Path
    output_path: Path | None = None
    final_output_path: Path | None = None
    summary_path: Path | None = None
    limit: int = DEFAULT_LIMIT
    resume: bool = False
    max_tokens: int = DEFAULT_MAX_TOKENS
    correct_field: str = "correct"
    id_field: str = "id"
    question_field: str = "question"
    model_final_answer_field: str = "model_final_answer"
    model_sympy_answer_fields: tuple[str, ...] = ("model_sympy_answer", "predicted_sympy_answer", "sympy_answer")
    reference_sympy_answer_fields: tuple[str, ...] = (
        "ground_truth_sympy_answer",
        "reference_sympy_answer",
        "sympy_answer",
    )
    progress_logs: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "input_path", Path(self.input_path))
        if self.output_path is not None:
            object.__setattr__(self, "output_path", Path(self.output_path))
        if self.final_output_path is not None:
            object.__setattr__(self, "final_output_path", Path(self.final_output_path))
        if self.summary_path is not None:
            object.__setattr__(self, "summary_path", Path(self.summary_path))
        object.__setattr__(self, "model_sympy_answer_fields", tuple(self.model_sympy_answer_fields))
        object.__setattr__(self, "reference_sympy_answer_fields", tuple(self.reference_sympy_answer_fields))


SYSTEM_INSTRUCTIONS = """You are an expert mathematical answer judge in a benchmark validation experiment.

Context:
- We evaluate model predictions on research-level math problems from a benchmark.
- Each item already passed through an automated SymPy-based string/normalization checker.
- For this item the automated checker marked the prediction as INCORRECT ("correct": false).
- Your job is a second-stage validation: decide whether the predicted answer is nevertheless
  conceptually equivalent to the reference answer and should be credited as correct.

Important:
- Do NOT compare raw strings only. Judge mathematical equivalence of the answers as written
  in their SymPy-normalized forms.
- "yes" means: a competent mathematician would accept the prediction as answering the same question 
  with the same mathematical content (possibly different representation, ordering, packaging, 
  equivalent numeric forms, same solution set, equivalent encodings when the same values are recoverable).
- "no" means: the prediction genuinely differs in mathematical content from the reference
  (wrong value, missing/extra solutions, wrong multi-part structure, materially different numeric result).
- If the difference appears to be only notation, packaging, conventional normalization, or a harmless neutral factor,
  lean "yes" with lower confidence. If the difference may change the mathematical object, lean "no".
- Use the question text to interpret what is being asked (counts, dimensions, solution sets, etc.).
- Judge up to the standard equivalence relation for that object unless the question explicitly asks for a literal presentation or a specific normal form.

Return exactly:
- explanation: one short sentence.
- verdict: "yes" or "no".
- confidence: integer 1-5.
"""


class LangChainJudge:
    """Single structured-output LangChain judge."""

    def __init__(self, spec: JudgeSpec, model: BaseChatModel) -> None:
        self.spec = spec
        self.model = model
        self.structured_model = model.with_structured_output(JudgeVerdict, **structured_output_kwargs(model))

    def judge(self, row: dict[str, Any], config: JudgeConfig) -> dict[str, Any]:
        messages = build_judge_messages(row, config)
        started = time.perf_counter()
        response = self.structured_model.invoke(messages)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        parsed, parse_error = _extract_structured_verdict(response)
        raw_message = response.get("raw") if isinstance(response, dict) else None
        raw_response = _raw_message_text(raw_message)
        usage = _usage_from_message(raw_message)

        return {
            "model": self.spec.label,
            "provider": self.spec.provider,
            "model_name": self.spec.model_name,
            "explanation": parsed.get("explanation"),
            "verdict": parsed.get("verdict"),
            "confidence": parsed.get("confidence"),
            "raw_response": raw_response,
            "parse_error": parse_error,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "elapsed_ms": elapsed_ms,
        }


def instantiate_judges(specs: Sequence[JudgeSpec], *, max_tokens: int) -> list[LangChainJudge]:
    return [
        LangChainJudge(spec=spec, model=instantiate_model_from_config(spec.provider, spec.model_name, max_tokens=max_tokens))
        for spec in specs
    ]


def instantiate_model_from_config(provider: str, model_name: str, *, max_tokens: int) -> BaseChatModel:
    """Instantiate a LangChain model from configs/model/{provider}.yaml."""

    config_dir = Path(hu.to_absolute_path(str(DEFAULT_MODEL_CONFIG_DIR)))
    base_path = config_dir / "base.yaml"
    provider_path = config_dir / f"{provider}.yaml"
    if not provider_path.exists():
        available = ", ".join(path.stem for path in sorted(config_dir.glob("*.yaml")) if path.stem != "base")
        raise ValueError(f"Unknown model config {provider!r}. Available configs: {available}")

    model_cfg = OmegaConf.merge(
        _load_model_yaml(base_path) if base_path.exists() else {},
        _load_model_yaml(provider_path),
    )
    if "defaults" in model_cfg:
        del model_cfg["defaults"]
    if max_tokens > 0:
        model_cfg["max_tokens"] = max_tokens

    cfg = OmegaConf.create({"model_name": model_name, "model": model_cfg})
    OmegaConf.resolve(cfg)
    model = hu.instantiate(cfg.model, _convert_="all")
    if not isinstance(model, BaseChatModel):
        raise TypeError(f"Config {provider!r} produced {type(model).__name__}, expected a LangChain chat model.")
    return model


def _load_model_yaml(path: Path) -> DictConfig:
    payload = OmegaConf.load(path)
    if not isinstance(payload, DictConfig):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def parse_judge_spec(value: str) -> JudgeSpec:
    """Parse 'provider=model_name' or 'label@provider=model_name'."""

    if "=" not in value:
        raise ValueError(f"Invalid judge spec {value!r}; expected 'provider=model_name', for example 'openai=gpt-5.5'.")
    left, model_name = value.split("=", 1)
    if not left.strip() or not model_name.strip():
        raise ValueError(f"Invalid judge spec {value!r}; provider and model_name must be non-empty.")

    if "@" in left:
        label, provider = left.split("@", 1)
        label = label.strip()
        provider = provider.strip()
    else:
        provider = left.strip()
        label = f"{provider}:{model_name.strip()}"

    if not provider:
        raise ValueError(f"Invalid judge spec {value!r}; provider must be non-empty.")
    if not label:
        label = f"{provider}:{model_name.strip()}"

    return JudgeSpec(provider=provider, model_name=model_name.strip(), label=label)


def load_judge_specs(values: Sequence[str] | None) -> list[JudgeSpec]:
    raw_values = list(values or [])
    if not raw_values:
        env_value = os.environ.get(DEFAULT_ENV_JUDGE_SPECS, "")
        raw_values = [item.strip() for item in env_value.split(",") if item.strip()]
    return [parse_judge_spec(value) for value in raw_values]


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value).strip()


def _resolve_first_present(row: dict[str, Any], field_names: Sequence[str]) -> tuple[Any, str]:
    for field_name in field_names:
        if field_name in row:
            return row[field_name], field_name
    return "", ""


def build_judge_messages(row: dict[str, Any], config: JudgeConfig) -> list[BaseMessage]:
    return [
        SystemMessage(content=SYSTEM_INSTRUCTIONS),
        HumanMessage(content=build_row_payload(row, config)),
    ]


def build_row_payload(row: dict[str, Any], config: JudgeConfig) -> str:
    reference_value, _ = _resolve_first_present(row, config.reference_sympy_answer_fields)
    model_value, _ = _resolve_first_present(row, config.model_sympy_answer_fields)
    payload = {
        "problem_id": row.get(config.id_field, ""),
        "question": str(row.get(config.question_field, "")),
        "ground_truth_sympy_answer": reference_value if reference_value is not None else "",
        "model_sympy_answer": model_value if model_value is not None else "",
    }
    if config.model_final_answer_field in row:
        payload["model_final_answer"] = str(row.get(config.model_final_answer_field, ""))
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _extract_structured_verdict(response: Any) -> tuple[dict[str, Any], str | None]:
    if not isinstance(response, dict):
        return {}, f"unexpected structured-output response type: {type(response).__name__}"

    parsing_error = response.get("parsing_error")
    parsed = response.get("parsed")
    if isinstance(parsed, JudgeVerdict):
        return parsed.model_dump(), str(parsing_error) if parsing_error else None
    if isinstance(parsed, dict):
        try:
            return JudgeVerdict.model_validate(parsed).model_dump(), str(parsing_error) if parsing_error else None
        except Exception as exc:  # noqa: BLE001 - keep parse failures in the output row
            return parsed, f"{exc.__class__.__name__}: {exc}"

    raw_text = _raw_message_text(response.get("raw"))
    parsed_from_text, text_error = parse_verdict(raw_text)
    if text_error is None:
        return parsed_from_text, str(parsing_error) if parsing_error else None
    return parsed_from_text, str(parsing_error or text_error)


def parse_verdict(raw: str) -> tuple[dict[str, Any], str | None]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"raw_response": raw}, f"JSONDecodeError: {exc}"
    if not isinstance(payload, dict):
        return {"raw_response": raw}, "top-level JSON value is not an object"
    try:
        return JudgeVerdict.model_validate(payload).model_dump(), None
    except Exception as exc:  # noqa: BLE001 - validation details are useful in output
        return payload, f"{exc.__class__.__name__}: {exc}"


def _raw_message_text(message: Any) -> str:
    if isinstance(message, AIMessage):
        content = message.content
    else:
        content = getattr(message, "content", message)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    if content is None:
        return ""
    return str(content)


def _usage_from_message(message: Any) -> Usage:
    usage_metadata = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") if isinstance(response_metadata, dict) else None
    token_usage = token_usage if isinstance(token_usage, dict) else {}

    prompt_tokens = usage_metadata.get("input_tokens") or token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
    completion_tokens = usage_metadata.get("output_tokens") or token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
    return Usage(prompt_tokens=int(prompt_tokens), completion_tokens=int(completion_tokens))


def load_results(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        rows: list[dict[str, Any]] = []
        for json_path in sorted(path.glob("*.json")):
            for row in _load_rows_file(json_path):
                copied = dict(row)
                copied.setdefault("_source_file", str(json_path))
                rows.append(copied)
        return rows
    return _load_rows_file(path)


def _load_rows_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return _load_jsonl_rows(path)
    return _load_json_rows(path)


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _load_object_stream_rows(text)

    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [row for row in payload["results"] if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("problems"), list):
        return [row for row in payload["problems"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _load_object_stream_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        try:
            payload, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            starts = _record_start_offsets(text)
            if not starts:
                raise
            return _load_rows_from_start_offsets(text, starts)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _load_rows_from_start_offsets(text: str, starts: Sequence[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if not chunk:
            continue
        payload = json.loads(chunk)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


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


def filter_incorrect(rows: list[dict[str, Any]], *, correct_field: str = "correct") -> list[dict[str, Any]]:
    return [row for row in rows if row.get(correct_field) is False]


def judge_row(
    row: dict[str, Any],
    judges: Sequence[LangChainJudge],
    config: JudgeConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verdicts: list[dict[str, Any]] = []
    for judge in judges:
        try:
            verdict = judge.judge(row, config)
        except Exception as exc:  # noqa: BLE001 - keep going on a single judge failure
            verdict = {
                "model": judge.spec.label,
                "provider": judge.spec.provider,
                "model_name": judge.spec.model_name,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        verdicts.append(verdict)
    return build_output_record(row, verdicts), verdicts


def aggregate_judges(judges: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid_verdicts = [
        str(judge.get("verdict")).lower()
        for judge in judges
        if judge.get("verdict") in {"yes", "no"} and not judge.get("parse_error") and not judge.get("error")
    ]
    yes_count = sum(1 for verdict in valid_verdicts if verdict == "yes")
    no_count = sum(1 for verdict in valid_verdicts if verdict == "no")
    total_valid = len(valid_verdicts)

    if total_valid == 0:
        majority = None
    elif yes_count > no_count:
        majority = "yes"
    elif no_count > yes_count:
        majority = "no"
    else:
        majority = "tie"

    return {
        "yes_votes": yes_count,
        "no_votes": no_count,
        "valid_votes": total_valid,
        "majority_verdict": majority,
        "majority_vote_correctness": majority == "yes",
    }


def build_output_record(row: dict[str, Any], judges: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge the original row with nested judge records and the majority vote."""

    record = dict(row)
    for index, judge in enumerate(judges, start=1):
        prefix = f"judge_{index}"
        record[prefix] = build_judge_record(judge)

    summary = aggregate_judges(judges)
    record.update(summary)
    return record


def build_judge_record(judge: dict[str, Any]) -> dict[str, Any]:
    record = {
        "model": judge.get("model"),
        "provider": judge.get("provider"),
        "model_name": judge.get("model_name"),
        "answer": {
            "explanation": judge.get("explanation"),
            "verdict": judge.get("verdict"),
            "confidence": judge.get("confidence"),
        },
        "raw_response": judge.get("raw_response"),
        "prompt_tokens": judge.get("prompt_tokens") or 0,
        "completion_tokens": judge.get("completion_tokens") or 0,
        "total_tokens": (judge.get("prompt_tokens") or 0) + (judge.get("completion_tokens") or 0),
        "elapsed_ms": judge.get("elapsed_ms"),
    }
    if judge.get("error"):
        record["error"] = judge.get("error")
    if judge.get("parse_error"):
        record["parse_error"] = judge.get("parse_error")
    return record


def drop_legacy_flat_judge_fields(record: dict[str, Any]) -> dict[str, Any]:
    """Remove pre-nested judge_k_* fields from older judge outputs."""

    return {key: value for key, value in record.items() if not LEGACY_FLAT_JUDGE_FIELD_RE.match(key)}


def load_existing_results(path: Path, *, id_field: str = "id") -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows = load_results(path)
    completed_ids = {_row_key(row, id_field=id_field) for row in rows}
    return rows, completed_ids


def save_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    temp_path.replace(path)


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    temp_path.replace(path)


def _row_key(row: dict[str, Any], *, id_field: str = "id") -> str:
    source = str(row.get("_source_file", ""))
    row_id = str(row.get(id_field, ""))
    return f"{source}::{row_id}" if source else row_id


def default_output_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    if input_path.is_dir():
        if input_path.name == "output":
            return input_path.parent / "judge" / "sympy_judge_verdicts.json"
        return input_path.parent / f"{input_path.name}_sympy_judge_verdicts.json"
    if input_path.parent.name == "output":
        return input_path.parent.parent / "judge" / input_path.name
    return input_path.parent / f"{input_path.stem}_sympy_judge_verdicts.json"


def default_summary_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    if input_path.is_dir():
        if input_path.name == "output":
            return input_path.parent / "judge" / "summary.json"
        return input_path.parent / f"{input_path.name}_judge_summary.json"
    if input_path.parent.name == "output":
        return input_path.parent.parent / "judge" / "summary.json"
    return input_path.parent / f"{input_path.stem}_judge_summary.json"


def default_final_output_path(input_path: Path) -> Path:
    input_path = Path(input_path)
    if input_path.is_dir():
        if input_path.name == "output":
            return input_path.parent / "final" / "sympy_final_votes.json"
        return input_path.parent / f"{input_path.name}_final_votes.json"
    if input_path.parent.name == "output":
        return input_path.parent.parent / "final" / input_path.name
    return input_path.parent / f"{input_path.stem}_final_votes.json"


def build_final_vote_rows(
    rows: Sequence[dict[str, Any]],
    judged_results: Sequence[dict[str, Any]],
    config: JudgeConfig,
) -> list[dict[str, Any]]:
    judged_by_key = {_row_key(row, id_field=config.id_field): row for row in judged_results}
    final_rows: list[dict[str, Any]] = []

    for row in rows:
        checker_correct = row.get(config.correct_field)
        if checker_correct is True:
            record = dict(row)
            record["final_vote"] = True
            record["final_vote_source"] = "symbolic_checker"
        elif checker_correct is False:
            judged_record = judged_by_key.get(_row_key(row, id_field=config.id_field))
            if judged_record is not None:
                record = dict(judged_record)
                record["final_vote"] = record.get("majority_vote_correctness") is True
                record["final_vote_source"] = "llm_judge_majority"
            else:
                record = dict(row)
                record["final_vote"] = False
                record["final_vote_source"] = "unjudged_checker_false"
        else:
            record = dict(row)
            record["final_vote"] = False
            record["final_vote_source"] = "missing_symbolic_checker_result"
        final_rows.append(drop_legacy_flat_judge_fields(record))

    return final_rows


def _truncate(value: Any, limit: int = CONSOLE_FIELD_LIMIT) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def run_judging(
    config: JudgeConfig,
    judges: Sequence[LangChainJudge],
) -> dict[str, Any]:
    if len(judges) != JUDGE_COUNT:
        raise ValueError(f"Expected exactly {JUDGE_COUNT} judges, got {len(judges)}")

    output_path = config.output_path or default_output_path(config.input_path)
    final_output_path = config.final_output_path or default_final_output_path(config.input_path)
    summary_path = config.summary_path or default_summary_path(config.input_path)
    rows = load_results(config.input_path)
    all_incorrect = filter_incorrect(rows, correct_field=config.correct_field)
    incorrect = all_incorrect[: config.limit] if config.limit and config.limit > 0 else all_incorrect

    results: list[dict[str, Any]] = []
    skipped = 0
    if config.resume:
        results, completed_ids = load_existing_results(output_path, id_field=config.id_field)
        before = len(incorrect)
        incorrect = [row for row in incorrect if _row_key(row, id_field=config.id_field) not in completed_ids]
        skipped = before - len(incorrect)

    _progress(config, f"Loaded {len(rows)} rows from {config.input_path}")
    _progress(
        config,
        f"Incorrect ({config.correct_field}=false): {len(all_incorrect)}; "
        f"to judge now: {len(incorrect)}" + (f" (skipped {skipped} already in output)" if skipped else ""),
    )
    _progress(config, "Judge models: " + ", ".join(judge.spec.label for judge in judges))

    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "majority_yes": 0,
        "majority_no": 0,
        "majority_tie": 0,
        "judge_errors": 0,
    }
    token_usage_by_model = {
        judge.spec.label: {
            "provider": judge.spec.provider,
            "model_name": judge.spec.model_name,
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        for judge in judges
    }

    for index, row in enumerate(incorrect, start=1):
        problem_id = row.get(config.id_field, f"row-{index:05d}")
        _progress(config, "_" * 80)
        _progress(config, f"Problem [{index}/{len(incorrect)}] id={problem_id}")
        _progress(config, f"\nQuestion:\n{_truncate(row.get(config.question_field))}")

        reference_value, _ = _resolve_first_present(row, config.reference_sympy_answer_fields)
        model_value, _ = _resolve_first_present(row, config.model_sympy_answer_fields)
        _progress(config, f"\nReference sympy:\n{_truncate(_format_value(reference_value))}")
        _progress(config, f"\nPredicted sympy:\n{_truncate(_format_value(model_value))}")

        record, verdicts = judge_row(row, judges, config)
        results.append(record)

        for judge_index, verdict in enumerate(verdicts, start=1):
            if verdict.get("error") or verdict.get("parse_error"):
                totals["judge_errors"] += 1
            totals["prompt_tokens"] += verdict.get("prompt_tokens") or 0
            totals["completion_tokens"] += verdict.get("completion_tokens") or 0
            model_label = str(verdict.get("model") or f"judge-{judge_index}")
            model_usage = token_usage_by_model.setdefault(
                model_label,
                {
                    "provider": verdict.get("provider"),
                    "model_name": verdict.get("model_name"),
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            )
            prompt_tokens = int(verdict.get("prompt_tokens") or 0)
            completion_tokens = int(verdict.get("completion_tokens") or 0)
            model_usage["calls"] = int(model_usage.get("calls") or 0) + 1
            model_usage["prompt_tokens"] = int(model_usage.get("prompt_tokens") or 0) + prompt_tokens
            model_usage["completion_tokens"] = int(model_usage.get("completion_tokens") or 0) + completion_tokens
            model_usage["total_tokens"] = int(model_usage.get("total_tokens") or 0) + prompt_tokens + completion_tokens
            _progress(
                config,
                f"\n  Judge {judge_index} ({verdict.get('model')}): "
                f"verdict={verdict.get('verdict')} confidence={verdict.get('confidence')} "
                f"tokens={verdict.get('prompt_tokens') or 0}+{verdict.get('completion_tokens') or 0} "
                f"- {_truncate(verdict.get('explanation') or verdict.get('error') or verdict.get('parse_error'))}",
            )

        if record.get("majority_vote_correctness") is True:
            totals["majority_yes"] += 1
        elif record.get("majority_verdict") == "no":
            totals["majority_no"] += 1
        else:
            totals["majority_tie"] += 1

        _progress(config, f"\nSummary: majority_vote_correctness={record.get('majority_vote_correctness')}")
        save_results(output_path, results)

    final_rows = build_final_vote_rows(rows, results, config)
    save_results(final_output_path, final_rows)
    final_vote_true = sum(1 for row in final_rows if row.get("final_vote") is True)

    summary = {
        "input_path": str(config.input_path),
        "output_path": str(output_path),
        "final_output_path": str(final_output_path),
        "summary_path": str(summary_path),
        "rows": len(rows),
        "incorrect_rows": len(all_incorrect),
        "judged_rows": len(incorrect),
        "skipped_rows": skipped,
        "final_vote_true": final_vote_true,
        "final_vote_false": len(final_rows) - final_vote_true,
        "judge_models": [judge.spec.label for judge in judges],
        "token_usage_by_model": token_usage_by_model,
        **totals,
    }
    save_summary(summary_path, summary)
    _progress(
        config,
        f"Done. majority_yes={totals['majority_yes']} majority_no={totals['majority_no']} "
        f"majority_tie={totals['majority_tie']} judge_errors={totals['judge_errors']} "
        f"prompt_tokens={totals['prompt_tokens']} completion_tokens={totals['completion_tokens']}",
    )
    _progress(config, f"Wrote {output_path}")
    _progress(config, f"Wrote judge summary to {summary_path}")
    _progress(config, f"Wrote final votes to {final_output_path}")
    return summary


def _progress(config: JudgeConfig, message: str) -> None:
    if config.progress_logs:
        print(message, flush=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run three LangChain LLM judges on benchmark rows where automated SymPy checking marked "
            "correct=false, then store a majority-vote correctness override."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Results JSON/JSONL file or directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write judge verdict rows. Defaults beside the input, or sibling judge/ for output/ files.",
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=None,
        help=("Path to write all original rows plus final_vote. Defaults beside the input, or sibling final/ for output/ files."),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Path to write aggregate judge logs. Defaults to summary.json in the judge folder.",
    )
    parser.add_argument(
        "--judge",
        action="append",
        default=[],
        help=(
            "Judge spec from configs/model/*.yaml. Use exactly three. Format: provider=model_name "
            "or label@provider=model_name, e.g. openai=gpt-5.5."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Max number of correct=false rows to judge (0 = all).",
    )
    parser.add_argument("--resume", action="store_true", help="Skip rows whose id is already in the output file.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Per-judge max output tokens.")
    parser.add_argument("--correct-field", default="correct", help="Boolean field used to select incorrect rows.")
    parser.add_argument("--id-field", default="id", help="Problem id field.")
    parser.add_argument("--question-field", default="question", help="Question text field.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-row console logging; final JSON summary is still printed.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        specs = load_judge_specs(args.judge)
        if len(specs) != JUDGE_COUNT:
            raise ValueError(
                f"Expected exactly {JUDGE_COUNT} judge specs, got {len(specs)}. "
                "Pass --judge provider=model_name three times or set "
                f"{DEFAULT_ENV_JUDGE_SPECS}=provider=model,provider=model,provider=model."
            )
        judges = instantiate_judges(specs, max_tokens=args.max_tokens)
        summary = run_judging(
            JudgeConfig(
                input_path=args.input,
                output_path=args.output,
                final_output_path=args.final_output,
                summary_path=args.summary,
                limit=args.limit,
                resume=args.resume,
                max_tokens=args.max_tokens,
                correct_field=args.correct_field,
                id_field=args.id_field,
                question_field=args.question_field,
                progress_logs=not args.quiet,
            ),
            judges,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should show a clear single error
        print(f"ERROR: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
