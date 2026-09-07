from dataclasses import asdict
import json

import pytest

from src.benchmark.lean_pipeline import (
    PROTOCOL, build_tasks, digest, render_prompt, summarize, verify_certificate,
)
from src.lean.types import LeanExecutionResult
from scripts.run_lean_verification import prepare_manifest, safe_config


def source(answer="2", **kwargs):
    return dict(id="p1", question="How much is one plus one?", model_sympy_answer=answer,
                model_final_answer=answer, explanation="Add the integers.", **kwargs)


def task():
    return build_tasks({"m": {"tool": [source(final_vote=True)]}})[0][0]


def test_full_file_labels_require_same_generation():
    row = source()
    labels = {"m": {"tool": [source("3", final_vote=False)]}}
    tasks, _ = build_tasks([row], labels, "m")
    assert tasks[0]["label"] is None
    labels["m"]["tool"] = [source(final_vote="False")]
    tasks, _ = build_tasks([row], labels, "m")
    assert tasks[0]["label"] is False


def test_no_tool_rows_missing_labels_and_error_strings():
    tasks, stats = build_tasks({"m": {"tool": [source(error="None")],
                                     "no_tool": [source(final_vote=True)]}})
    assert len(tasks) == 1 and tasks[0]["label"] is None
    assert stats["labels"] == {"None": 1}


def test_prompt_allowlist_and_context_ablation():
    row = task()
    row.update(theorem="SECRET_THEOREM", ground_truth="SECRET_REFERENCE", label_source="SECRET_JUDGE",
               candidate_explanation="EXPLANATION_SENTINEL", sage_steps=[{"code": "TRACE_SENTINEL"}])
    for context in ("answer", "explanation", "traces"):
        prompt = render_prompt(row, context)
        assert "SECRET" not in prompt
        assert ("EXPLANATION_SENTINEL" in prompt) == (context != "answer")
        assert ("TRACE_SENTINEL" in prompt) == (context == "traces")


def test_source_trace_string_is_parsed_and_failed_calls_retained():
    row = source(tool_traces=str([{"name": "sage_exec", "ok": False,
                                   "arguments": {"code": "bad()"}, "content": "error"}]))
    tasks, _ = build_tasks([row], family="m")
    assert tasks[0]["sage_steps"][0]["ok"] is False


def record(t, **kwargs):
    return dict(id=t["id"], task_sha256=digest(t), protocol=PROTOCOL,
                verdict="PROVED", lean_proof="final snippet", **kwargs)


def test_unrelated_success_cannot_count_and_pending_errors_stay_in_denominator():
    t = task()
    t2 = dict(t, id="m::p2", problem_id="p2", label=False)
    t3 = dict(t, id="m::p3", problem_id="p3")
    rows = [record(t, tool_traces=[{"name": "lean_exec", "metadata": {"proved": True}}]),
            record(t2, error={"type": "APIError"})]
    stats = summarize([t, t2, t3], rows)
    assert stats["all"]["kernel_proved"] == 0
    assert stats["all"]["n"] == 3
    assert stats["all"]["pending"] == stats["all"]["errors"] == 1


def test_audit_is_bound_to_exact_task_and_certificate():
    t = task()
    r = record(t, final_certificate={"checked": True, "proved": True,
                                     "code_sha256": digest("final snippet")})
    a = dict(id=t["id"], faithful=True, reviewer="human", task_sha256=digest(t), code_sha256="stale")
    assert summarize([t], [r], [a])["all"]["audited_proved"] == 0
    a["code_sha256"] = digest("final snippet")
    assert summarize([t], [r], [a])["all"]["audited_proved"] == 1


def test_resume_deduplication_and_input_mismatch():
    t = task()
    r = record(t)
    assert summarize([t], [record(t, error={"type": "APIError"}), r])["all"]["completed"] == 1
    with pytest.raises(ValueError, match="Multiple completed"):
        summarize([t], [r, r])
    with pytest.raises(ValueError, match="mismatch"):
        summarize([dict(t, question="changed")], [r])


def test_manifest_rejects_mixing(tmp_path):
    p, out = tmp_path / "manifest.json", tmp_path / "out.jsonl"
    prepare_manifest(p, {"model": "a"}, out, True)
    prepare_manifest(p, {"model": "a"}, out, True)
    with pytest.raises(ValueError, match="mismatch"):
        prepare_manifest(p, {"model": "b"}, out, True)
    assert safe_config({"model": {"api_key": "secret"}})["model"]["api_key"] == "<redacted>"


def test_final_check_targets_submitted_certificate():
    class Runtime:
        def execute_lean_code(self, code, decl_name):
            assert code.startswith("submitted code")
            assert "Not _root_.candidate_claim := _root_.candidate_certificate" in code
            assert decl_name == "pipeline_checked_certificate"
            return LeanExecutionResult(status="error", proved=False)
    result = verify_certificate(Runtime(), {"verdict": "REFUTED", "lean_proof": "submitted code"})
    assert result["checked"] and not result["proved"]
