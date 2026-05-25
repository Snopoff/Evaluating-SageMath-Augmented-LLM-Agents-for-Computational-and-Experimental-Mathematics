from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.benchmark.compare_predictions import _load_rows
from src.benchmark.judge_sympy_answers import load_results


def load_prediction_rows(path: Path, limit: int) -> tuple[list[dict[str, Any]], int]:
    if path.is_dir():
        rows = load_results(path)
        if limit != -1:
            rows = rows[:limit]
        return rows, 0
    return _load_rows(path, limit)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
