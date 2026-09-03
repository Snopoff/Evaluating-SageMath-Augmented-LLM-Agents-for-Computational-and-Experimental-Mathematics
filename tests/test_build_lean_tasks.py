"""Tests for the Lean task-dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from scripts.build_lean_tasks import (  # noqa: E402
    build_claims_audit,
    build_discrimination,
    build_problem_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DISCRIMINATION_PATH = REPO_ROOT / "data/processed/lean_discrimination.json"

_MINIMAL_PROBLEM = {
    "p1": {
        "id": "p1",
        "question": "q",
        "theorem": "t",
        "ground_truth_latex": "$1$",
        "ground_truth_sympy": "1",
        "ground_truth_is_multi": False,
    }
}


@pytest.fixture(scope="module")
def tasks():
    if not DISCRIMINATION_PATH.exists():
        pytest.skip("run scripts/build_lean_tasks.py first")
    with DISCRIMINATION_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_dataset_is_balanced_by_construction(tasks):
    """A discriminator that always answers the same way must score exactly 50%."""
    correct = [task for task in tasks if task["label"]]
    incorrect = [task for task in tasks if not task["label"]]
    assert len(correct) == len(incorrect)
    assert len(tasks) == 2 * len({task["problem_id"] for task in tasks})


def test_every_problem_contributes_exactly_one_of_each_label(tasks):
    seen: dict[str, set[bool]] = {}
    for task in tasks:
        seen.setdefault(task["problem_id"], set()).add(task["label"])
    assert all(labels == {True, False} for labels in seen.values())


def test_candidates_are_non_empty_and_differ_within_a_problem(tasks):
    by_problem: dict[str, dict[bool, str]] = {}
    for task in tasks:
        assert task["candidate_sympy"].strip(), task["id"]
        by_problem.setdefault(task["problem_id"], {})[task["label"]] = task["candidate_sympy"]
    for problem_id, pair in by_problem.items():
        assert pair[True] != pair[False], f"{problem_id} has identical candidates"


def test_problem_context_is_populated(tasks):
    """The agent needs the statement; an empty one would silently degrade the run."""
    for task in tasks:
        assert task["theorem"].strip(), task["id"]
        assert task["question"].strip(), task["id"]
        # Always text, even for multi-answer problems whose raw ground truth is a list.
        assert isinstance(task["ground_truth_sympy"], str), task["id"]
        assert task["ground_truth_sympy"].strip(), task["id"]


def test_multi_answer_problems_are_kept_and_flagged(tasks):
    multi = [task for task in tasks if task["ground_truth_is_multi"]]
    assert multi, "multi-answer problems must not be silently dropped"
    for task in multi:
        assert ";" in task["ground_truth_sympy"] or task["ground_truth_sympy"]


def test_contested_flag_tracks_label_source(tasks):
    for task in tasks:
        assert task["contested"] == (task["label_source"] == "llm_judge_majority")


def test_incorrect_candidates_differ_from_the_reference(tasks):
    for task in tasks:
        if not task["label"]:
            assert task["candidate_sympy"] != task["ground_truth_sympy"], task["id"]


def test_builder_requires_both_classes():
    """A problem answered the same way by everyone yields no task at all."""
    results = {
        "fam": {
            "tool": [
                {"id": "p1", "answer_type": "expression", "model_sympy_answer": "1", "final_vote": True},
                {"id": "p1", "answer_type": "expression", "model_sympy_answer": "2", "final_vote": True},
            ]
        }
    }
    assert build_discrimination(results, _MINIMAL_PROBLEM) == []


def test_builder_skips_rows_without_a_usable_expression():
    """Error/timeout rows carry an empty expression and cannot be adjudicated."""
    results = {
        "fam": {
            "tool": [
                {"id": "p1", "answer_type": "expression", "model_sympy_answer": "1", "final_vote": True},
                {"id": "p1", "answer_type": "expression", "model_sympy_answer": "", "final_vote": False},
            ]
        }
    }
    assert build_discrimination(results, _MINIMAL_PROBLEM) == []


def test_builder_pick_is_deterministic():
    results = {
        "b_fam": {"tool": [
            {"id": "p1", "answer_type": "expression", "model_sympy_answer": "9", "final_vote": True},
            {"id": "p1", "answer_type": "expression", "model_sympy_answer": "8", "final_vote": False},
        ]},
        "a_fam": {"tool": [
            {"id": "p1", "answer_type": "expression", "model_sympy_answer": "7", "final_vote": True},
            {"id": "p1", "answer_type": "expression", "model_sympy_answer": "6", "final_vote": False},
        ]},
    }
    first = build_discrimination(results, _MINIMAL_PROBLEM)
    second = build_discrimination(results, _MINIMAL_PROBLEM)
    assert first == second
    # Sorted by family name, so the alphabetically first family wins.
    assert all(task["source_family"] == "a_fam" for task in first)


def test_claims_audit_attaches_only_executed_sage_as_evidence():
    results = {
        "fam": {
            "tool": [
                {
                    "id": "p1",
                    "answer_type": "expression",
                    "verified_claims": ["genus equals 3(r-1)"],
                    "tool_traces": [
                        {"name": "sage_exec", "ok": True, "turn": 1,
                         "arguments": {"code": "RESULT = 3"},
                         "content": "3", "metadata": {"result_data": 3}},
                        {"name": "sage_exec", "ok": False, "turn": 2,
                         "arguments": {"code": "boom"}, "content": "err", "metadata": {}},
                        {"name": "query-docs", "ok": True, "turn": 3,
                         "arguments": {}, "content": "docs", "metadata": {}},
                    ],
                    "final_vote": True,
                }
            ]
        }
    }
    audit = build_claims_audit(results, _MINIMAL_PROBLEM)
    assert len(audit) == 1
    assert audit[0]["claim"] == "genus equals 3(r-1)"
    # Failed calls and documentation lookups are not evidence.
    assert [step["code"] for step in audit[0]["sage_steps"]] == ["RESULT = 3"]


def test_problem_index_joins_both_files():
    index = build_problem_index()
    assert index, "problem index must not be empty"
    sample = next(iter(index.values()))
    assert set(sample) >= {"id", "question", "theorem", "ground_truth_latex", "ground_truth_sympy"}
