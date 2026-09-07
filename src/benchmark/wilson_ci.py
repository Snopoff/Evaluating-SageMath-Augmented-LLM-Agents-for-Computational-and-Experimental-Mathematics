"""Wilson score intervals for binomial proportions.

Reconstructed: `scripts/wilson_ci.py` imports this module, but only a stale
`__pycache__` entry was left on disk, so that CLI currently fails with
ModuleNotFoundError. The signatures below match what the bytecode exposed.

The Wilson interval is used rather than the normal approximation because our
subgroup counts are small (a single arXiv category can hold two problems), and
the normal interval misbehaves badly near 0 and 1 -- exactly where the coverage
and unsoundness numbers sit.
"""

import argparse
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any

DEFAULT_CORRECT_FIELDS = ("correct", "is_correct", "matches_reference")

#: Two-sided normal quantiles for the confidence levels we actually use.
_Z_BY_LEVEL = {0.80: 1.2815515655, 0.90: 1.6448536270, 0.95: 1.9599639845, 0.99: 2.5758293035}


def _z_for(confidence_level: float) -> float:
    try:
        return _Z_BY_LEVEL[round(confidence_level, 2)]
    except KeyError as exc:
        supported = ", ".join(str(level) for level in sorted(_Z_BY_LEVEL))
        raise ValueError(f"Unsupported confidence_level {confidence_level!r}; supported: {supported}") from exc


@dataclass(frozen=True)
class WilsonSummary:
    correct: int
    rows: int
    accuracy: float
    confidence_level: float
    lower_bound: float
    upper_bound: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def wilson_interval(correct: int, rows: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for `correct` successes out of `rows` trials."""
    if rows < 0 or correct < 0:
        raise ValueError("correct and rows must be non-negative.")
    if correct > rows:
        raise ValueError("correct cannot exceed rows.")
    if rows == 0:
        return 0.0, 0.0

    z = _z_for(confidence_level)
    n = float(rows)
    phat = correct / n
    denominator = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denominator
    margin = (z / denominator) * sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def wilson_summary(correct: int, rows: int, confidence_level: float = 0.95) -> WilsonSummary:
    lower, upper = wilson_interval(correct, rows, confidence_level)
    return WilsonSummary(
        correct=correct,
        rows=rows,
        accuracy=(correct / rows) if rows else 0.0,
        confidence_level=confidence_level,
        lower_bound=lower,
        upper_bound=upper,
    )


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in handle if line.strip()]
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("problems") or payload.get("rows") or []
    return list(payload)


def summarize_result_rows(
    *,
    input_paths: Sequence[Path],
    correct_fields: Iterable[str] = DEFAULT_CORRECT_FIELDS,
    limit: int | None = None,
    id_filter_paths: Sequence[Path] | None = None,
    confidence_level: float = 0.95,
) -> WilsonSummary:
    correct_fields = tuple(correct_fields)

    keep_ids: set[str] | None = None
    if id_filter_paths:
        keep_ids = set()
        for path in id_filter_paths:
            for row in _load_rows(Path(path)):
                if isinstance(row, dict) and "id" in row:
                    keep_ids.add(row["id"])
                elif isinstance(row, str):
                    keep_ids.add(row)

    rows: list[dict[str, Any]] = []
    for path in input_paths:
        rows.extend(_load_rows(Path(path)))
    if keep_ids is not None:
        rows = [row for row in rows if row.get("id") in keep_ids]
    if limit is not None and limit >= 0:
        rows = rows[:limit]

    scored = 0
    correct = 0
    for row in rows:
        for field in correct_fields:
            if field in row:
                scored += 1
                correct += bool(row[field])
                break
    return wilson_summary(correct, scored, confidence_level)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--correct", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--input-path", type=Path, action="append", default=[])
    parser.add_argument(
        "--correct-field",
        action="append",
        default=[],
        help=f"Defaults to {DEFAULT_CORRECT_FIELDS}. Pass final_vote for judge-adjusted labels.",
    )
    parser.add_argument("--id-filter-path", type=Path, action="append", default=[])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    args = parser.parse_args()

    if args.correct is not None and args.rows is not None:
        summary = wilson_summary(args.correct, args.rows, args.confidence_level)
    elif args.input_path:
        summary = summarize_result_rows(
            input_paths=args.input_path,
            correct_fields=tuple(args.correct_field) or DEFAULT_CORRECT_FIELDS,
            limit=args.limit,
            id_filter_paths=args.id_filter_path,
            confidence_level=args.confidence_level,
        )
    else:
        parser.error("provide either --correct/--rows or at least one --input-path")

    print(json.dumps(summary.as_dict(), indent=2))


if __name__ == "__main__":
    main()
