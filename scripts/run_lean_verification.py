"""Run the Lean verification agent over a prebuilt task file.

Concurrency model: **one** Lean worker, several agent threads.

`import Mathlib` costs ~35 s and ~6 GB of RSS, but once warm each `lean_exec`
call returns in milliseconds, while every LLM turn takes tens of seconds. So the
Lean process is never the bottleneck and a single shared runtime serves all
threads; spawning one runtime per thread would only multiply the memory bill.

`GeneratePredictionsRunner` is not reused here because it is strictly sequential
and has no resume.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
import hydra.utils as hu
import rootutils
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.agent.schemas import LeanFinalAnswerArgs  # noqa: E402
from src.tools.catalog import LEAN_EXEC_TOOL_NAME, make_lean_exec_tool  # noqa: E402
from src.utils.config_helpers import resolve_text_asset  # noqa: E402
from src.benchmark.lean_pipeline import (  # noqa: E402
    PROTOCOL, digest, render_prompt, verify_certificate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def render_discrimination_prompt(task: dict[str, Any]) -> str:
    """Compose the formalize-and-prove prompt.

    Only `question` and the candidate go in. The task record also carries a
    `theorem` field -- the statement as written in the source paper -- and that
    field **must never be shown to the agent**: in this dataset it states the
    theorem together with its answer, so including it hands over the ground
    truth and turns the task into a comparison against the reference instead of
    an independent formalization. It is kept in the record for offline auditing
    of faithfulness only.
    """
    parts = [
        "# Problem",
        task.get("question", "").strip(),
        "",
        "# Candidate answer to formalize and prove",
        "Another agent obtained this by computing examples in a computer algebra",
        "system and guessing the pattern. It is a conjecture, not a known fact:",
        "",
        "```",
        task.get("candidate_sympy", "").strip(),
        "```",
    ]
    raw = task.get("candidate_raw", "").strip()
    if raw:
        parts += ["", f"As the other agent originally wrote it: {raw}"]

    # The exploration the other agent already did. Given as a reference point for
    # the Lean definitions, not as an invitation to redo the search.
    steps = task.get("sage_steps") or []
    if steps:
        parts += [
            "",
            "# What the other agent computed to get there",
            "Use these to check your own definitions reproduce the same values.",
            "Do not redo this exploration.",
            "",
        ]
        for step in steps:
            parts += [
                f"Sage, turn {step.get('turn')}:",
                "```python",
                str(step.get("code", "")),
                "```",
                "gave:",
                "```",
                str(step.get("output", "")),
                "```",
                "",
            ]
    parts += [
        "Define the objects in Lean, reconcile them with the computations above,",
        "state the candidate over your definitions, and prove it.",
    ]
    return "\n".join(parts)


def render_claims_prompt(task: dict[str, Any]) -> str:
    """Compose the claim-audit prompt, including the Sage output behind the claim."""
    parts = [
        "# Problem context",
        # `question`, never `theorem`: the latter states the answer alongside the
        # claim, which would hand the auditor the conclusion it is meant to check.
        task.get("question", "").strip(),
        "",
        "# Claim to audit",
        "An agent asserted the following, saying it was supported by its computation:",
        "",
        f"> {task.get('claim', '').strip()}",
    ]
    steps = task.get("sage_steps") or []
    if steps:
        parts += ["", "# What the agent actually computed"]
        for step in steps[:4]:
            parts += [
                f"Sage code (turn {step.get('turn')}):",
                "```python",
                str(step.get("code", ""))[:1500],
                "```",
                "Output:",
                "```",
                str(step.get("output", ""))[:800],
                "```",
                "",
            ]
    parts += [
        "Decide whether this claim is true, using Lean 4 and Mathlib.",
        "Formalize only the checkable mathematical content of the claim.",
        "If the claim is too vague to state formally, return UNKNOWN with failure_kind 'missing-concept'.",
    ]
    return "\n".join(parts)


def load_tasks(path: Path, start_with: int, limit: int) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    rows = rows[start_with:]
    if limit is not None and limit >= 0:
        rows = rows[:limit]
    return rows


def completed_ids(output_path: Path) -> set[str]:
    """Ids already *successfully* written, so an interrupted run can resume.

    Records that failed with an exception (an API 402 when credits run out, a
    rate limit, a timeout) are deliberately not counted as done: otherwise a
    transient outage would silently drop those tasks from the sample for good,
    and the run would look complete while missing them.
    """
    if not output_path.exists():
        return set()
    done: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("error"):
            continue
        if "id" in record:
            done.add(record["id"])
    return done


def safe_config(value: Any) -> Any:
    """Record configuration without credentials, even with literal overrides."""
    if isinstance(value, dict):
        return {k: ("<redacted>" if any(s in k.lower() for s in ("api_key", "password", "secret", "access_token"))
                    else safe_config(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_config(v) for v in value]
    return value


def prepare_manifest(path: Path, manifest: dict, output_path: Path, resume: bool) -> None:
    """Never silently mix datasets, prompts, models, or protocol versions."""
    if output_path.exists() and not resume:
        raise ValueError("Output exists and resume=false; choose a new run_name")
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Run manifest mismatch; choose a new run_name")
    elif output_path.exists():
        raise ValueError("Existing results have no manifest; choose a new run_name")
    else:
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


@hydra.main(version_base=None, config_path="../configs", config_name="lean_verification")
def main(cfg: DictConfig) -> None:
    settings = cfg.lean_verification
    logger = hu.instantiate(cfg.logger, mode="lean_verification")
    logger.setup_logging()

    tasks_path = Path(hu.to_absolute_path(str(settings.tasks_path)))
    output_dir = Path(hu.to_absolute_path(str(settings.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{settings.run_name}.jsonl"

    protocol = str(settings.get("protocol", "legacy"))
    is_pipeline = protocol == PROTOCOL
    is_claims_run = "claims" in tasks_path.name
    render = (lambda task: render_prompt(task, str(settings.context))) if is_pipeline else (
        render_claims_prompt if is_claims_run else render_discrimination_prompt
    )

    tasks = load_tasks(tasks_path, int(settings.start_with), int(settings.limit))
    if len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    if is_pipeline and any(t.get("protocol") != PROTOCOL for t in tasks):
        raise ValueError("Use the new pipeline builder to create tasks for this protocol")
    lean_usage_notes = (
        resolve_text_asset(cfg.lean_skill, label="lean_skill", logger=logger)
        if cfg.get("lean_skill") is not None else ""
    )
    system_prompt = resolve_text_asset(cfg.system_prompt, label="system_prompt", logger=logger)
    if is_pipeline:
        configuration = safe_config(OmegaConf.to_container(cfg, resolve=False))
        configuration["lean_verification"].pop("dry_run", None)
        configuration["lean_verification"].pop("resume", None)
        configuration["lean"] = OmegaConf.to_container(cfg.lean, resolve=True)
        source_paths = [Path(__file__), REPO_ROOT / "src/benchmark/lean_pipeline.py",
                        REPO_ROOT / "src/lean/runtime.py", REPO_ROOT / "src/lean/types.py",
                        REPO_ROOT / "src/agent/controller.py", REPO_ROOT / "src/agent/schemas.py",
                        REPO_ROOT / "src/tools/catalog.py", REPO_ROOT / "lean_ws/lean-toolchain",
                        REPO_ROOT / "lean_ws/lake-manifest.json", REPO_ROOT / "uv.lock"]
        manifest = {"protocol": PROTOCOL, "tasks_sha256": digest(tasks),
                    "configuration": configuration, "system_prompt": system_prompt,
                    "lean_usage_notes": lean_usage_notes,
                    "source_hashes": {str(p.relative_to(REPO_ROOT)): digest(p.read_text())
                                      for p in source_paths if p.exists()}}
        manifest["run_sha256"] = digest(manifest)
        prepare_manifest(output_path.with_suffix(".manifest.json"), manifest, output_path, bool(settings.resume))
        if output_path.exists():
            for line in output_path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)  # Refuse damaged tails instead of appending invalid JSONL.
                    if record.get("run_sha256") != manifest["run_sha256"]:
                        raise ValueError("Result record has a different run manifest")
        if bool(settings.get("dry_run", False)):
            preview = output_path.with_suffix(".inputs.jsonl")
            with preview.open("w") as out:
                for task in tasks:
                    out.write(json.dumps({"id": task["id"], "task_sha256": digest(task),
                                          "user_prompt": render(task)}, ensure_ascii=False) + "\n")
            logger.progress(f"Dry run: {len(tasks)} exact user inputs -> {preview}; no model or Lean started")
            return
    already_done = completed_ids(output_path) if bool(settings.resume) else set()
    pending = [task for task in tasks if task["id"] not in already_done]

    logger.progress(
        f"{len(tasks)} tasks, {len(already_done)} already done, {len(pending)} to run "
        f"-> {output_path}"
    )
    if not pending:
        return

    # One runtime, shared by every thread. This is the whole memory story.
    lean_runtime = hu.instantiate(cfg.lean, logger=logger)

    write_lock = threading.Lock()
    handle = output_path.open("a", encoding="utf-8")

    def solve(task: dict[str, Any]) -> dict[str, Any]:
        # A controller per thread: the shared piece is the Lean runtime, which
        # serialises internally, not the agent state.
        started = time.monotonic()
        user_prompt = render(task)
        try:
            model = hu.instantiate(cfg.model, _convert_="all")
            controller = hu.instantiate(
                cfg.controller, model=model,
                tools=[make_lean_exec_tool(lean_runtime, lean_usage_notes)],
                logger=logger, system_prompt=system_prompt, model_name=cfg.model_name,
                final_answer_schema=LeanFinalAnswerArgs, exec_tool_names=[LEAN_EXEC_TOOL_NAME],
            )
            result = controller.solve(user_prompt)
            payload = result.final_payload or {}
            record = {
                "id": task["id"],
                "verdict": payload.get("verdict", ""),
                "failure_kind": payload.get("failure_kind", ""),
                "lean_statement": payload.get("lean_statement", ""),
                "lean_proof": payload.get("lean_proof", ""),
                "explanation": result.explanation,
                "confidence": result.confidence,
                "verified_claims": result.verified_claims,
                "tool_traces": result.tool_traces,
                "turn_count": result.turn_count,
                "stop_reason": result.stop_reason,
                "token_usage": result.token_usage,
                "error": None,
            }
            if is_pipeline:
                record["final_certificate"] = verify_certificate(lean_runtime, payload)
        except Exception as exc:
            record = {
                "id": task["id"],
                "verdict": "",
                "failure_kind": "",
                "stop_reason": "exception",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        # Carry the label through untouched so analysis needs no second join.
        for field in (
            "problem_id", "label", "label_source", "contested", "arxiv_tag",
            "candidate_sympy", "ground_truth_sympy", "ground_truth_is_multi",
            "source_family", "source_arm", "claim", "answer_was_correct",
        ):
            if field in task:
                record[field] = task[field]
        record["solve_time_sec"] = round(time.monotonic() - started, 3)
        if is_pipeline:
            record.update(protocol=PROTOCOL, task_sha256=digest(task),
                          run_sha256=manifest["run_sha256"], context=str(settings.context),
                          user_prompt=user_prompt)
        return record

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=int(settings.max_workers)) as pool:
            futures = {pool.submit(solve, task): task for task in pending}
            for future in as_completed(futures):
                record = future.result()
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                completed += 1
                logger.progress(
                    f"[{completed}/{len(pending)}] {record['id']} -> "
                    f"{record.get('verdict') or record.get('stop_reason')}"
                )
    finally:
        handle.close()
        lean_runtime.shutdown()

    logger.progress(f"done: {completed} records appended to {output_path}")


if __name__ == "__main__":
    main()
