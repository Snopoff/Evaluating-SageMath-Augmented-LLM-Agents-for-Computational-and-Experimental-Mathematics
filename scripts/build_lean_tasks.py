"""Build the Lean verification datasets from the existing Sage-stage logs.

No new agent runs: everything here is a join over artefacts already on disk.

Track A+B (`lean_discrimination`): for every expression problem that some
configuration answered correctly and some other configuration answered
incorrectly, emit one correct and one incorrect candidate. The result is
balanced 50/50 by construction, so a discriminator that always says the same
thing scores 50%.

Track C (`lean_claims_audit`): the agents' own `verified_claims`, paired with the
Sage output that supposedly backs them.

Labels come from `final_vote`, not `correct`: `correct` is the symbolic checker
alone, while `final_vote` is the label the previous paper actually reported.
`final_vote_source` records which of the two decided, which is what lets us
report the contested slice separately.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = REPO_ROOT / "data/results/all_results_aaai.json"
# Statements live in per-answer-type files; the union is needed because the
# pipeline slice keeps numeric answers too ("prove the answer is 42").
PROBLEMS_PATHS = (
    REPO_ROOT / "data/processed/problems_by_answer_type_expression.json",
    REPO_ROOT / "data/processed/problems_by_answer_type_number.json",
    REPO_ROOT / "data/processed/problems_by_answer_type_formula.json",
)
PROBLEMS_PATH = PROBLEMS_PATHS[0]
NORMALIZED_PATH = REPO_ROOT / "data/processed/normalized_problems.json"

TRIAGE_PATH = REPO_ROOT / "data/processed/lean_formalizability.json"

DISCRIMINATION_OUT = REPO_ROOT / "data/processed/lean_discrimination.json"
CLAIMS_OUT = REPO_ROOT / "data/processed/lean_claims_audit.json"


def as_expression_text(value: Any) -> tuple[str, bool]:
    """Normalize a ground truth to text, flagging the multi-answer case.

    Problems with `answer_kind == "multi"` store a list of expressions. They are
    kept rather than dropped -- excluding them would bias the set toward the
    easiest answer shapes -- but analysis needs to be able to split them out.
    """
    if isinstance(value, list):
        return " ; ".join(str(item).strip() for item in value if str(item).strip()), True
    return str(value or "").strip(), False


def reference_visible_in(text: str, reference_latex: str) -> bool:
    """Does the reference answer appear verbatim in text the agent will see?

    Used as an audit flag, not a filter. `theorem` states the answer and is never
    shown; `question` occasionally contains a short answer string like `2^k`
    incidentally as part of the setup, which is harmless but worth marking.
    """
    def squash(value: str) -> str:
        return re.sub(r"[\s{}\\$()\[\]]", "", value or "").lower()

    needle = squash((reference_latex or "").strip("$"))
    if len(needle) < 3:
        return False
    return needle in squash(text)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_problem_index() -> dict[str, dict[str, Any]]:
    """Join the two problem files into one record per id.

    Neither file alone is sufficient: only `problems_by_answer_type_expression`
    carries `theorem` and the enriched question, and only the normalized file
    carries the SymPy ground truth. The results files cannot substitute for
    either -- their `ground_truth` field is empty for every expression record.
    """
    problems: dict[str, Any] = {}
    for path in PROBLEMS_PATHS:
        if not path.exists():
            continue
        for row in load_json(path):
            problems.setdefault(row["id"], row)
    normalized = {row["id"]: row for row in load_json(NORMALIZED_PATH)}

    index: dict[str, dict[str, Any]] = {}
    for problem_id, row in problems.items():
        llm = row.get("llm") or {}
        norm = normalized.get(problem_id, {})
        ground_truth_sympy, is_multi = as_expression_text(norm.get("sympy_answer", ""))
        index[problem_id] = {
            "id": problem_id,
            "question": llm.get("revised_question") or row.get("question", ""),
            "original_question": row.get("question", ""),
            "theorem": row.get("theorem", ""),
            "ground_truth_latex": row.get("answer", ""),
            "ground_truth_sympy": ground_truth_sympy,
            "ground_truth_is_multi": is_multi,
        }
    return index


def iter_expression_rows(results: dict[str, Any], expression_only: bool = True):
    for family, arms in results.items():
        for arm, rows in arms.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if expression_only and row.get("answer_type") != "expression":
                    continue
                yield family, arm, row


def sage_evidence(row: dict[str, Any], max_steps: int = 4) -> list[dict[str, Any]]:
    """The numerical work the Sage agent did before guessing this answer.

    Passed to the Lean stage as a reference for its own definitions: the Lean
    agent is not asked to re-explore the problem (that already happened here),
    only to check that whatever it defines reproduces these values before
    trying to prove the candidate over it.
    """
    steps: list[dict[str, Any]] = []
    for trace in row.get("tool_traces") or []:
        if trace.get("name") != "sage_exec" or not trace.get("ok"):
            continue
        steps.append(
            {
                "turn": trace.get("turn"),
                "code": (trace.get("arguments") or {}).get("code", "")[:1500],
                "output": str(trace.get("content", ""))[:800],
            }
        )
    return steps[:max_steps]


def candidate_expression(row: dict[str, Any]) -> str:
    for field in ("model_sympy_answer", "prediction_sympy_answer", "sympy_answer"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_discrimination(
    results: dict[str, Any],
    problems: dict[str, dict[str, Any]],
    tool_arms_only: bool = True,
):
    """One correct and one incorrect candidate per eligible problem."""
    by_problem: dict[str, dict[bool, list[dict[str, Any]]]] = defaultdict(
        lambda: {True: [], False: []}
    )

    for family, arm, row in iter_expression_rows(results):
        # Only candidates from Sage-enabled runs. The pipeline under study is
        # "Sage explores numerically, then Lean proves what it found", so a
        # candidate produced by a tool-free model never went through stage 1
        # and does not belong in the set.
        if tool_arms_only and arm != "tool":
            continue
        expression = candidate_expression(row)
        if not expression:
            continue
        label = bool(row.get("final_vote"))
        by_problem[row["id"]][label].append(
            {
                "family": family,
                "arm": arm,
                "candidate_sympy": expression,
                "candidate_raw": row.get("model_final_answer", ""),
                "label_source": row.get("final_vote_source", ""),
                "match_type": row.get("match_type", ""),
                "symbolic_correct": bool(row.get("correct")),
                "arxiv_tag": row.get("arxiv_tag", ""),
                "sage_steps": sage_evidence(row),
            }
        )

    tasks: list[dict[str, Any]] = []
    for problem_id in sorted(by_problem):
        buckets = by_problem[problem_id]
        if not buckets[True] or not buckets[False]:
            continue  # need both classes to make a balanced pair
        problem = problems.get(problem_id)
        if problem is None:
            continue

        for label in (True, False):
            # Deterministic pick: sort by (family, arm, expression) and take the
            # first, so re-running the builder always yields the same dataset.
            chosen = sorted(
                buckets[label],
                key=lambda c: (c["family"], c["arm"], c["candidate_sympy"]),
            )[0]
            tasks.append(
                {
                    "id": f"{problem_id}::{'correct' if label else 'incorrect'}",
                    "problem_id": problem_id,
                    "question": problem["question"],
                    "theorem": problem["theorem"],
                    "ground_truth_latex": problem["ground_truth_latex"],
                    "ground_truth_sympy": problem["ground_truth_sympy"],
                    "ground_truth_is_multi": problem.get("ground_truth_is_multi", False),
                    "reference_visible_in_question": reference_visible_in(
                        problem["question"], problem["ground_truth_latex"]
                    ),
                    "candidate_sympy": chosen["candidate_sympy"],
                    "candidate_raw": chosen["candidate_raw"],
                    "label": label,
                    "label_source": chosen["label_source"],
                    # The contested slice: the symbolic checker said "wrong" and an
                    # LLM judge overturned it. This is where formal verification
                    # would add the most, and where the prior paper is weakest.
                    "contested": chosen["label_source"] == "llm_judge_majority",
                    "symbolic_correct": chosen["symbolic_correct"],
                    "match_type": chosen["match_type"],
                    "source_family": chosen["family"],
                    "source_arm": chosen["arm"],
                    "arxiv_tag": chosen["arxiv_tag"],
                    "sage_steps": chosen["sage_steps"],
                }
            )
    return tasks


def build_claims_audit(results: dict[str, Any], problems: dict[str, dict[str, Any]]):
    """Self-reported `verified_claims`, paired with the Sage output behind them."""
    audit: list[dict[str, Any]] = []
    for family, arm, row in iter_expression_rows(results):
        claims = row.get("verified_claims") or []
        if not isinstance(claims, list) or not claims:
            continue
        problem = problems.get(row["id"])
        if problem is None:
            continue

        # Recover what Sage actually computed, so the audit compares the claim
        # against the computation rather than against the claim's own wording.
        sage_steps = []
        for trace in row.get("tool_traces") or []:
            if trace.get("name") != "sage_exec" or not trace.get("ok"):
                continue
            metadata = trace.get("metadata") or {}
            sage_steps.append(
                {
                    "turn": trace.get("turn"),
                    "code": (trace.get("arguments") or {}).get("code", ""),
                    "output": str(trace.get("content", ""))[:2000],
                    "result_data": metadata.get("result_data"),
                }
            )

        for index, claim in enumerate(claims):
            if not isinstance(claim, str) or not claim.strip():
                continue
            audit.append(
                {
                    "id": f"{row['id']}::{family}::{arm}::{index}",
                    "problem_id": row["id"],
                    "claim": claim.strip(),
                    "question": problem["question"],
                    "theorem": problem["theorem"],
                    "sage_steps": sage_steps,
                    "source_family": family,
                    "source_arm": arm,
                    "answer_was_correct": bool(row.get("final_vote")),
                    "arxiv_tag": row.get("arxiv_tag", ""),
                }
            )
    return audit


def sample_claims(claims: list[dict[str, Any]], per_family: int) -> list[dict[str, Any]]:
    """Deterministic stratified sample: equal quota per model family.

    Sampling by family rather than at random keeps one prolific configuration
    from dominating the audit, which would turn a claim about "CAS agents" into
    a claim about one model.
    """
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        by_family[claim["source_family"]].append(claim)

    sampled: list[dict[str, Any]] = []
    for family in sorted(by_family):
        rows = sorted(by_family[family], key=lambda c: c["id"])
        if len(rows) <= per_family:
            sampled.extend(rows)
            continue
        # Even stride over the sorted list: spreads the sample across problems
        # instead of taking a contiguous block of the same few problems.
        stride = len(rows) / per_family
        sampled.extend(rows[int(index * stride)] for index in range(per_family))
    return sampled


def build_pipeline_slice(
    results: dict[str, Any],
    problems: dict[str, dict[str, Any]],
    *,
    triage_path: Path = TRIAGE_PATH,
    classes: tuple[str, ...] = ("mathlib-native",),
    max_per_paper: int = 4,
    include_number_answers: bool = True,
) -> list[dict[str, Any]]:
    """Tasks for the "prove the candidate" pipeline, sliced by formalizability.

    Two differences from `build_discrimination`:

    * Answer type is not filtered. "Prove that the answer is 42" is a perfectly
      good Lean goal -- often an easier one than a symbolic identity -- and the
      expression-only restriction was inherited from the discriminator framing.
    * A per-source-paper cap. Problems are heavily clustered by source: 16 of the
      21 `mathlib-native` math.NT problems come from a single arXiv paper, so
      without a cap a "sample of 21" is really a sample of about 5 independent
      settings and the measured rate would describe one paper.
    """
    triage = {row["id"]: row for row in load_json(triage_path)["results"]}

    by_problem: dict[str, dict[bool, list[dict[str, Any]]]] = defaultdict(
        lambda: {True: [], False: []}
    )
    meta: dict[str, dict[str, Any]] = {}
    for family, arm, row in iter_expression_rows(results, expression_only=not include_number_answers):
        if arm != "tool":
            continue
        expression = candidate_expression(row)
        if not expression:
            continue
        meta.setdefault(row["id"], {"arxiv_id": row.get("arxiv_id", ""),
                                    "arxiv_tag": row.get("arxiv_tag", "")})
        by_problem[row["id"]][bool(row.get("final_vote"))].append(
            {
                "family": family,
                "arm": arm,
                "candidate_sympy": expression,
                "candidate_raw": row.get("model_final_answer", ""),
                "label_source": row.get("final_vote_source", ""),
                "answer_type": row.get("answer_type", ""),
                "sage_steps": sage_evidence(row),
            }
        )

    eligible = [
        problem_id
        for problem_id in sorted(by_problem)
        if problem_id in problems
        and triage.get(problem_id, {}).get("formalizability") in classes
    ]

    # Cap per source paper so one prolific paper cannot dominate the slice.
    per_paper: Counter = Counter()
    selected: list[str] = []
    for problem_id in eligible:
        paper = meta.get(problem_id, {}).get("arxiv_id") or problem_id
        if per_paper[paper] >= max_per_paper:
            continue
        per_paper[paper] += 1
        selected.append(problem_id)

    tasks: list[dict[str, Any]] = []
    for problem_id in selected:
        problem = problems[problem_id]
        info = triage.get(problem_id, {})
        for label in (True, False):
            bucket = by_problem[problem_id][label]
            if not bucket:
                continue
            chosen = sorted(bucket, key=lambda c: (c["family"], c["candidate_sympy"]))[0]
            tasks.append(
                {
                    "id": f"{problem_id}::{'correct' if label else 'incorrect'}",
                    "problem_id": problem_id,
                    "question": problem["question"],
                    "theorem": problem["theorem"],
                    "ground_truth_latex": problem["ground_truth_latex"],
                    "ground_truth_sympy": problem["ground_truth_sympy"],
                    "candidate_sympy": chosen["candidate_sympy"],
                    "candidate_raw": chosen["candidate_raw"],
                    "sage_steps": chosen["sage_steps"],
                    "label": label,
                    "label_source": chosen["label_source"],
                    "contested": chosen["label_source"] == "llm_judge_majority",
                    "answer_type": chosen["answer_type"],
                    "formalizability": info.get("formalizability", ""),
                    "checkable_at_instances": info.get("checkable_at_instances"),
                    "required_concepts": info.get("required_concepts", []),
                    "source_family": chosen["family"],
                    "source_arm": chosen["arm"],
                    "arxiv_tag": meta.get(problem_id, {}).get("arxiv_tag", ""),
                    "arxiv_id": meta.get(problem_id, {}).get("arxiv_id", ""),
                }
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    parser.add_argument("--discrimination-out", type=Path, default=DISCRIMINATION_OUT)
    parser.add_argument("--claims-out", type=Path, default=CLAIMS_OUT)
    parser.add_argument("--slice-classes", nargs="*", default=None,
                        help="Build a pipeline slice for these formalizability classes.")
    parser.add_argument("--slice-out", type=Path,
                        default=REPO_ROOT / "data/processed/lean_slice.json")
    parser.add_argument("--max-per-paper", type=int, default=4)
    parser.add_argument(
        "--claims-sample-per-family",
        type=int,
        default=0,
        help="If set, also write a stratified sample with this many claims per model family.",
    )
    args = parser.parse_args()

    results = load_json(args.results_path)
    problems = build_problem_index()

    discrimination = build_discrimination(results, problems)
    claims = build_claims_audit(results, problems)

    args.discrimination_out.write_text(
        json.dumps(discrimination, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.claims_out.write_text(
        json.dumps(claims, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.claims_sample_per_family > 0:
        sample = sample_claims(claims, args.claims_sample_per_family)
        sample_path = args.claims_out.with_name(args.claims_out.stem + "_sample.json")
        sample_path.write_text(
            json.dumps(sample, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"claims sample:  {len(sample)} claims over "
              f"{len({c['source_family'] for c in sample})} families -> {sample_path}")

    if args.slice_classes:
        sliced = build_pipeline_slice(
            results, problems,
            classes=tuple(args.slice_classes),
            max_per_paper=args.max_per_paper,
        )
        args.slice_out.write_text(
            json.dumps(sliced, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        papers = len({t["arxiv_id"] for t in sliced})
        print(f"slice ({', '.join(args.slice_classes)}): {len(sliced)} tasks over "
              f"{len({t['problem_id'] for t in sliced})} problems from {papers} papers "
              f"-> {args.slice_out}")

    n_correct = sum(1 for task in discrimination if task["label"])
    n_contested = sum(1 for task in discrimination if task["contested"])
    print(f"discrimination: {len(discrimination)} tasks "
          f"({n_correct} correct / {len(discrimination) - n_correct} incorrect), "
          f"{len({t['problem_id'] for t in discrimination})} problems, "
          f"{n_contested} contested -> {args.discrimination_out}")
    print(f"claims audit:   {len(claims)} claims "
          f"over {len({c['problem_id'] for c in claims})} problems -> {args.claims_out}")


if __name__ == "__main__":
    main()
