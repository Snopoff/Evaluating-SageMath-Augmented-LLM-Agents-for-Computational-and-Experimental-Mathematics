# Sage answers → Lean certificates

This protocol attempts to prove complete answers produced by the Sage agent.
It is an offline second stage over saved logs, with no feedback into Sage.
The main comparison is proof acceptance for judge-correct versus judge-incorrect
answers under the same Lean model, prompt, and tool budget.

## Build the dataset

From the repository root, with the project environment installed:

```bash
.venv/bin/python -m src.benchmark.lean_pipeline build \
  --results data/results/all_results_current.json \
  --out data/processed/lean_pipeline.json
```

Only `tool` arms are used, across all answer types. No balancing, requirement
that both labels occur, or formalizability filter is applied. Empty answers and
source errors are excluded and counted in the dataset manifest. The question is
the exact question recorded in the Sage log, not the source theorem containing
the reference answer. The original answer, normalized answer, explanation, and
Sage tool calls are preserved. Tool calls may themselves have failed.

To use one raw log and join its judge labels:

```bash
.venv/bin/python -m src.benchmark.lean_pipeline build \
  --results data/results/deepseek-v4-pro/tool/full.json \
  --labels data/results/all_results_current.json \
  --family deepseek-v4-pro \
  --out data/processed/lean_pipeline_deepseek_v4_pro.json
```

The family must match the aggregate key. A label join requires identical problem
ID, question, candidate, and original answer. Missing or ambiguous labels stay
unknown; they are never converted to false. A `final_vote` already in the source
row is used directly. Judge decisions are labels, not guaranteed mathematical truth.

For a small pilot add `--limit 8`; for a single family from the aggregate add
`--families deepseek-v4-pro`. Selection uses a fixed seed (default 2026), without
looking at labels. Use the same saved task file for every context ablation.

## Inspect and run

The existing Lean runtime needs the workspace's pinned Lean/Mathlib installation
and REPL binary (`configs/lean/default.yaml`). Model credentials and endpoint are
configured through the existing model config/environment. The inherited default
model is `anthropic/claude-opus-4.5`; change `model_name` explicitly if needed.

```bash
.venv/bin/python scripts/run_lean_verification.py --config-name lean_pipeline \
  lean_verification.run_name=pilot8_explanation \
  lean_verification.tasks_path=data/processed/lean_pipeline_pilot8.json \
  lean_verification.dry_run=true
```

The dry run writes exact user prompts and a manifest without starting Lean or
calling a model. Build the pilot task file first using `build --limit 8 --out ...`.
Repeat the command with `lean_verification.dry_run=false` to execute it.

Available `lean_verification.context` values:

| Context | Input from Sage |
|---|---|
| `answer` | Question, normalized and original answer |
| `explanation` (default) | Above, plus final explanation |
| `traces` | Above, plus all recorded Sage code and outputs, including failure flags |

There is no silent truncation of Sage evidence. Large traces can exceed a model's
context window; those failures are recorded. Use distinct `run_name` values for
different contexts, models, or budgets. The default controller budget is 12 steps
and two finalization retries; Lean calls have a 60-second timeout. Final replay
is an additional Lean call outside the agent's tool budget. Four agent threads
share one Lean runtime. Source logs are data, not instructions to the Lean agent.

Manifests bind the selected tasks, prompts, config, source hashes, lockfile, and
Lean workspace versions. User prompts are also stored in every result. Resume
skips completed attempts and retries exceptions. A changed manifest or malformed
result file is rejected. Use a new run name after code or protocol changes.
Keep the original JSONL files and manifests; never concatenate different runs.

## What counts as a proof

The final snippet must define `candidate_claim : Prop` and a root declaration
`candidate_certificate : candidate_claim` for PROVED, or
`candidate_certificate : ¬ candidate_claim` for REFUTED. The runner appends a
typed wrapper and checks the final snippet in a fresh base environment, including
its axioms. Intermediate successful tool calls do not count. Neither a failed
`decide` nor a verbal counterexample counts as a refutation.

This binds a certificate to a formal proposition. It does **not** establish that
the proposition faithfully translates the natural-language question. In
particular, `candidate_claim := True`, a missing quantifier, assumed conjecture,
or a conveniently chosen definition still requires semantic review.

The report therefore separates:

1. Agent-claimed verdicts.
2. Independently replayed kernel certificates (`kernel_proved`, `kernel_refuted`).
3. Certificates with a completed faithfulness review (`audited_proved`, `audited_refuted`).

Generate a review template without exposing judge labels:

```bash
.venv/bin/python -m src.benchmark.lean_pipeline audit-template \
  --tasks data/processed/lean_pipeline_pilot8.json \
  --results outputs/lean_verification/pilot8_explanation.jsonl \
  --out outputs/lean_verification/pilot8_audit.json
```

A reviewer fills `faithful` with true/false, `reviewer`, and `notes`. Check all
definitions, hypotheses, domains, quantifiers, non-vacuity, and the full requested
conclusion (e.g. uniqueness or completeness of a solution set). Resolve any
conflict with a judge label against the actual mathematics. Reviews are tied to
the exact task and code hashes; stale reviews cannot count.

## Analyze

```bash
.venv/bin/python -m src.benchmark.lean_pipeline analyze \
  --tasks data/processed/lean_pipeline_pilot8.json \
  --results outputs/lean_verification/pilot8_explanation.jsonl \
  --audits outputs/lean_verification/pilot8_audit.json \
  --out outputs/lean_verification/pilot8_summary.json
```

Omit `--audits` before review. The report uses the entire selected task manifest
as the denominator, with errors and pending records explicit. It also gives
rates among completed attempts. Results are split by label and Sage family;
unlabeled answers stay separate. Repeated answers to the same problem from
different models are dependent observations, so there is no naive binomial CI
over the pooled rows. For a paper, report unique problem counts and use paired
comparisons or problem-clustered uncertainty when comparing contexts.

The principal table should show n, errors/pending, UNKNOWN, kernel proofs, audited
proofs and refutations, separately for each label. `UNKNOWN` is abstention, not a
negative answer. The experiment tests whether proof acceptance is a useful
selective correctness signal; finite-budget proof failure cannot supply an
oracle of falsity. A claim of replacing the judge requires both high coverage
on correct answers and independently audited avoidance of false acceptance.

Do not pool legacy `trackB_mini` results with this protocol: their input and
certificate-counting rules differ.

## Local validation

```bash
.venv/bin/python -m pytest tests/test_lean_pipeline.py tests/test_build_lean_tasks.py -q
.venv/bin/python -m pytest tests/test_lean_runtime.py -k pipeline_final_certificate -q
```

The second command starts real Lean. It covers a complete proof, a negation,
an unrelated successful lemma, `sorry`, and an invented axiom.
