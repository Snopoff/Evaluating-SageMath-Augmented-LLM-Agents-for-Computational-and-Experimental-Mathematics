from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from src.benchmark.components.config import BenchmarkConfig
from src.benchmark.sample_unused_problems_by_arxiv_category import (
    _load_tags_by_id,
    _required_categories,
)
from src.benchmark.split_predictions_by_answer_type import (
    ANSWER_TYPES,
    _answer_type_from_row,
    _load_answer_types_by_id,
)


def load_tags_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return _load_tags_by_id(path)


def load_answer_types_by_id(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return _load_answer_types_by_id(path)


def enrich_rows(
    rows: Sequence[dict[str, Any]],
    *,
    config: BenchmarkConfig,
    tags_by_id: dict[str, dict[str, Any]],
    answer_types_by_id: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enriched_rows: list[dict[str, Any]] = []
    missing_tag_ids: list[str] = []
    missing_answer_type_ids: list[str] = []
    arxiv_tag_counts: Counter[str] = Counter()
    answer_type_counts: Counter[str] = Counter()

    for index, row in enumerate(rows):
        enriched = dict(row)
        problem_id = row.get(config.id_field)
        if not isinstance(problem_id, str) or not problem_id:
            problem_id = f"row-{index + 1:05d}"

        tag_row = tags_by_id.get(problem_id)
        if tag_row is None:
            missing_tag_ids.append(problem_id)
            enriched.setdefault("arxiv_tag", "")
            enriched.setdefault("arxiv_categories", [])
        else:
            categories = _required_categories(tag_row)
            arxiv_tag = categories[0]
            enriched["arxiv_tag"] = arxiv_tag
            enriched["arxiv_categories"] = categories
            if "arxiv_id" in tag_row:
                enriched["arxiv_id"] = tag_row.get("arxiv_id")
            if "link" in tag_row:
                enriched["arxiv_link"] = tag_row.get("link")
            arxiv_tag_counts[arxiv_tag] += 1

        answer_type = _answer_type_from_row(enriched) or answer_types_by_id.get(problem_id, "")
        if answer_type in ANSWER_TYPES:
            enriched["answer_type"] = answer_type
            answer_type_counts[answer_type] += 1
        else:
            missing_answer_type_ids.append(problem_id)
            enriched.setdefault("answer_type", "")

        enriched_rows.append(enriched)

    if config.strict_metadata:
        errors = []
        if missing_tag_ids:
            errors.append(_format_missing("arXiv tags", missing_tag_ids))
        if missing_answer_type_ids:
            errors.append(_format_missing("answer types", missing_answer_type_ids))
        if errors:
            raise ValueError("; ".join(errors))

    return enriched_rows, {
        "missing_arxiv_tag_count": len(missing_tag_ids),
        "missing_arxiv_tag_ids": missing_tag_ids,
        "missing_answer_type_count": len(missing_answer_type_ids),
        "missing_answer_type_ids": missing_answer_type_ids,
        "arxiv_tag_counts": dict(sorted(arxiv_tag_counts.items())),
        "answer_type_counts": dict(sorted(answer_type_counts.items())),
    }


def _format_missing(label: str, problem_ids: Sequence[str]) -> str:
    preview = ", ".join(problem_ids[:10])
    suffix = "" if len(problem_ids) <= 10 else f", ... ({len(problem_ids)} total)"
    return f"missing {label} for: {preview}{suffix}"
