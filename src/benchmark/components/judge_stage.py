from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from src.benchmark.components.config import BenchmarkConfig
from src.benchmark.components.io import write_json
from src.benchmark.judge_sympy_answers import (
    JUDGE_COUNT,
    JudgeConfig,
    LangChainJudge,
    build_final_vote_rows,
    instantiate_judges,
    load_judge_specs,
    load_results,
    run_judging,
    save_results,
)


def run_judge_stage(
    *,
    config: BenchmarkConfig,
    sympy_output_path: Path,
    final_output_path: Path,
    judge_output_path: Path,
    judge_summary_path: Path,
    incorrect_rows: int,
    judges: Sequence[LangChainJudge] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    judge_config = JudgeConfig(
        input_path=sympy_output_path,
        output_path=judge_output_path,
        final_output_path=final_output_path,
        summary_path=judge_summary_path,
        limit=config.judge_limit,
        resume=config.judge_resume,
        max_tokens=config.judge_max_tokens,
        correct_field=config.correct_field,
        id_field=config.id_field,
        question_field=config.question_field,
        progress_logs=config.progress_logs,
    )

    if incorrect_rows == 0:
        rows = load_results(sympy_output_path)
        final_rows = build_final_vote_rows(rows, [], judge_config)
        save_results(judge_output_path, [])
        save_results(final_output_path, final_rows)
        summary = {
            "input_path": str(sympy_output_path),
            "output_path": str(judge_output_path),
            "final_output_path": str(final_output_path),
            "summary_path": str(judge_summary_path),
            "rows": len(rows),
            "incorrect_rows": 0,
            "judged_rows": 0,
            "skipped_rows": 0,
            "final_vote_true": len(final_rows),
            "final_vote_false": 0,
            "judge_models": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "majority_yes": 0,
            "majority_no": 0,
            "majority_tie": 0,
            "judge_errors": 0,
            "token_usage_by_model": {},
        }
        write_json(judge_summary_path, summary)
        if progress is not None:
            progress("judge stage skipped because SymPy marked every row correct")
        return summary

    resolved_judges = resolve_judges(config=config, judges=judges)
    if progress is not None:
        progress(f"judge stage processing {incorrect_rows} SymPy-false rows")
    return run_judging(judge_config, resolved_judges)


def resolve_judges(
    *,
    config: BenchmarkConfig,
    judges: Sequence[LangChainJudge] | None = None,
) -> list[LangChainJudge]:
    if judges is not None:
        resolved = list(judges)
        if len(resolved) != JUDGE_COUNT:
            raise ValueError(f"Expected exactly {JUDGE_COUNT} judges, got {len(resolved)}")
        return resolved

    specs = load_judge_specs(config.judge_specs)
    if len(specs) != JUDGE_COUNT:
        raise ValueError(
            f"SymPy marked rows as false, so LLM judging requires exactly {JUDGE_COUNT} "
            f"judge specs; got {len(specs)}. Set benchmark.config.judge_specs in "
            "configs/benchmark.yaml, e.g. [openai=gpt-5.5, anthropic=claude-opus-4-7, "
            "google=gemini-3.5-flash], or set JUDGE_SPECS to the same comma-separated list."
        )
    return instantiate_judges(specs, max_tokens=config.judge_max_tokens)
