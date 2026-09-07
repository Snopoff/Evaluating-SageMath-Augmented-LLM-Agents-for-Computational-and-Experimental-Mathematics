"""Metrics for the Lean verification runs (tracks A, B and C).

Track B asks whether formal verifiability is a usable correctness signal for
CAS-agent answers. The dataset is balanced 50/50 by construction, so a constant
predictor scores exactly 50% and any signal above that is real.

Three numbers carry the result:

- **coverage** -- how often Lean reached a verdict at all;
- **accuracy on covered** -- whether those verdicts matched the reference label;
- **unsoundness** -- verdicts that were confidently wrong. This is the number
  that decides whether formal verification can be trusted as an answer checker,
  and it is reported separately rather than folded into accuracy.

Everything is also reported on the *contested* slice: the items where the
symbolic checker said "wrong" and an LLM judge overturned it. That slice is
where the previous paper's headline is weakest and where a formal check would
add the most.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.benchmark.wilson_ci import wilson_summary  # noqa: E402

DECIDED = ("PROVED", "REFUTED")

# A goal of `True` (or `0 = 0`) is provable and says nothing about the candidate.
VACUOUS_GOAL_RE = re.compile(r":\s*True\s*:=|:\s*(\d+)\s*=\s*\1\s*:=")

# Uninterpreted symbols: `axiom`/`opaque`/`constant`/bare `variable`. A theorem
# about an arbitrary opaque function is not a theorem about the object in the
# problem, so a proof of it certifies nothing. This stays a red flag.
SELF_DECLARED_RE = re.compile(r"^\s*(variables?|axiom|opaque|constant)\b", re.M)

# Definitions with actual bodies. The agent is *asked* to write these -- the
# problems' objects are not in Mathlib -- so this is a capability statistic,
# not a warning. Faithfulness of these definitions is audited separately.
OWN_DEF_RE = re.compile(r"^\s*(noncomputable\s+)?(def|abbrev|inductive|structure)\b", re.M)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def has_vacuous_goal(row: dict[str, Any]) -> bool:
    return bool(VACUOUS_GOAL_RE.search(row.get("lean_statement") or ""))


def has_self_declared_symbols(row: dict[str, Any]) -> bool:
    """Did the agent axiomatize the very objects it was asked to reason about?

    Note this is intentionally *not* the same as token overlap between the
    candidate and the statement: the candidate is written in SymPy notation and
    the statement in Lean, so a faithful formalization routinely shares no
    identifiers at all.
    """
    return bool(SELF_DECLARED_RE.search(row.get("lean_statement") or ""))


def count_own_definitions(row: dict[str, Any]) -> int:
    """How many objects the agent had to define itself to state the problem."""
    return len(OWN_DEF_RE.findall(row.get("lean_proof") or row.get("lean_statement") or ""))


def is_suspect_formalization(row: dict[str, Any]) -> bool:
    """Diagnostic only -- never used to override a verdict."""
    return has_vacuous_goal(row) or has_self_declared_symbols(row)


def is_machine_backed(row: dict[str, Any]) -> bool:
    """Did at least one `lean_exec` call in this run actually return proved=True?

    A verdict is only as good as the check behind it. Agents sometimes announce
    PROVED/REFUTED after computing a counterexample in their own head and
    writing it into a comment, leaving the Lean snippet on `sorry`. Such a
    verdict may still be right, but nothing was verified -- so claimed coverage
    and machine-backed coverage are reported as two different numbers.
    """
    for trace in row.get("tool_traces") or []:
        if trace.get("name") != "lean_exec":
            continue
        if (trace.get("metadata") or {}).get("proved"):
            return True
    return False


def verdict_is_correct(row: dict[str, Any]) -> bool | None:
    """None when Lean abstained; otherwise whether the verdict matched the label."""
    verdict = row.get("verdict")
    if verdict not in DECIDED:
        return None
    return (verdict == "PROVED") == bool(row.get("label"))


def summarize(rows: Iterable[dict[str, Any]], name: str) -> dict[str, Any]:
    rows = [row for row in rows if not row.get("error")]
    decided = [row for row in rows if row.get("verdict") in DECIDED]
    right = [row for row in decided if verdict_is_correct(row)]
    wrong = [row for row in decided if verdict_is_correct(row) is False]

    backed = [row for row in decided if is_machine_backed(row)]
    coverage = wilson_summary(len(decided), len(rows))
    backed_coverage = wilson_summary(len(backed), len(rows))
    accuracy = wilson_summary(len(right), len(decided))
    unsound = wilson_summary(len(wrong), len(rows))
    return {
        "slice": name,
        "n": len(rows),
        "coverage_claimed": coverage.as_dict(),
        "coverage_machine_backed": backed_coverage.as_dict(),
        "unbacked_verdicts": len(decided) - len(backed),
        "accuracy_on_covered": accuracy.as_dict(),
        "unsoundness_over_all": unsound.as_dict(),
        "verdicts": dict(Counter(row.get("verdict") or "ERROR" for row in rows)),
        "vacuous_goals": sum(1 for row in rows if has_vacuous_goal(row)),
        "self_declared_symbols": sum(1 for row in rows if has_self_declared_symbols(row)),
    }


def fmt(summary: dict[str, Any]) -> str:
    def pct(entry: dict[str, Any]) -> str:
        if not entry["rows"]:
            return "     n/a      "
        return (f"{entry['accuracy']*100:5.1f}% "
                f"[{entry['lower_bound']*100:4.1f},{entry['upper_bound']*100:5.1f}]")

    return (
        f"{summary['slice']:<26s} n={summary['n']:<4d} "
        f"cover {pct(summary['coverage_claimed'])}  "
        f"backed {pct(summary['coverage_machine_backed'])}  "
        f"acc {pct(summary['accuracy_on_covered'])}  "
        f"unsound {pct(summary['unsoundness_over_all'])}"
    )


def report_discrimination(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]
    failed = [row for row in rows if row.get("error")]

    print(f"records: {len(rows)}  valid: {len(valid)}  errored: {len(failed)}")
    if failed:
        print("  error types:", dict(Counter(r["error"]["type"] for r in failed)))
    print()

    slices = {
        "all": valid,
        "contested (judge)": [r for r in valid if r.get("contested")],
        "uncontested (sympy)": [r for r in valid if not r.get("contested")],
        "label=correct": [r for r in valid if r.get("label")],
        "label=incorrect": [r for r in valid if not r.get("label")],
    }
    summaries = {name: summarize(subset, name) for name, subset in slices.items()}
    print("=== Track B: Lean as a discriminator ===")
    for summary in summaries.values():
        print(" ", fmt(summary))

    print("\n=== Track A: why the pipeline stops ===")
    kinds = Counter(
        row.get("failure_kind") or "(none)"
        for row in valid
        if row.get("verdict") == "UNKNOWN"
    )
    unknown_total = sum(kinds.values())
    for kind, count in kinds.most_common():
        share = count / unknown_total * 100 if unknown_total else 0
        print(f"  {kind:20s} {count:4d}  ({share:5.1f}% of UNKNOWN)")

    print("\n=== by arXiv category ===")
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_category[row.get("arxiv_tag") or "?"].append(row)
    for category, subset in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        if len(subset) < 2:
            continue
        print(" ", fmt(summarize(subset, category)))

    vacuous = [row for row in valid if has_vacuous_goal(row)]
    declared = [row for row in valid if has_self_declared_symbols(row)]
    print("\n=== verdict backing ===")
    decided_all = [row for row in valid if row.get("verdict") in DECIDED]
    unbacked = [row for row in decided_all if not is_machine_backed(row)]
    print(f"  decided verdicts:                  {len(decided_all)}")
    print(f"  backed by a machine-checked proof: {len(decided_all) - len(unbacked)}")
    print(f"  asserted without any Lean proof:   {len(unbacked)}")
    for row in unbacked:
        print(f"    {row['id']} [{row['verdict']}]")

    print("\n=== formalization faithfulness (diagnostics, not verdict overrides) ===")
    defined = [row for row in valid if count_own_definitions(row)]
    print(f"  wrote its own Lean definitions:          {len(defined)}/{len(valid)}"
          f"  (mean {sum(count_own_definitions(r) for r in defined)/max(1,len(defined)):.1f} defs)")
    print(f"  vacuous goals (True / n = n):            {len(vacuous)}/{len(valid)}")
    print(f"  used opaque/axiomatized symbols:         {len(declared)}/{len(valid)}")
    decided_and_suspect = [
        row for row in valid
        if row.get("verdict") in DECIDED and is_suspect_formalization(row)
    ]
    print(f"  ...of which reached a PROVED/REFUTED verdict: {len(decided_and_suspect)}")
    for row in decided_and_suspect[:5]:
        print(f"    {row['id']} [{row['verdict']}]: "
              f"{(row.get('lean_statement') or '')[:80]}")

    print("\n=== effort ===")
    turns = [row.get("turn_count", 0) for row in valid]
    tokens = [(row.get("token_usage") or {}).get("total_tokens", 0) for row in valid]
    times = [row.get("solve_time_sec", 0) for row in valid]
    if valid:
        print(f"  mean turns {sum(turns)/len(valid):.1f} | "
              f"mean tokens {sum(tokens)//len(valid):,} | "
              f"mean wall {sum(times)/len(valid):.0f}s | "
              f"total tokens {sum(tokens):,}")
    return summaries


def report_claims(rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if not row.get("error")]
    print(f"=== Track C: audit of self-reported verified_claims ===")
    print(f"records: {len(rows)}  valid: {len(valid)}")
    verdicts = Counter(row.get("verdict") or "ERROR" for row in valid)
    for verdict, count in verdicts.most_common():
        print(f"  {verdict:10s} {count:5d}  ({count/max(1,len(valid))*100:5.1f}%)")
    refuted = [row for row in valid if row.get("verdict") == "REFUTED"]
    print(f"\n  claims the agent called verified but Lean refuted: {len(refuted)}")
    for row in refuted[:10]:
        print(f"    [{row.get('source_family')}] {(row.get('claim') or '')[:110]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+", help="one or more .jsonl runs")
    parser.add_argument("--claims", action="store_true", help="treat input as a track C run")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.results:
        rows.extend(load_rows(path))
    # Later runs win, so a re-run of a failed shard supersedes the failure.
    deduped = {row["id"]: row for row in rows}
    rows = list(deduped.values())

    if args.claims:
        report_claims(rows)
        return

    summaries = report_discrimination(rows)
    if args.json_out:
        args.json_out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
