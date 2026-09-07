# Agent guide for LLMxM2

## Scope and project purpose

This guide applies throughout this repository. It was prepared for the `lean`
branch; recheck the current branch and working tree before editing. Do not switch
branches or discard existing work merely to match this document.

The project (Python package metadata name `Sagent`) evaluates LLM agents on
computational and experimental mathematics, comparing SageMath-assisted answers
with answers generated without tools. This branch also implements an offline
Sage-answer-to-Lean-certificate experiment. Its goal is to measure whether formal
proof acceptance is a useful selective correctness signal, not to treat failure
to find a proof as evidence that an answer is false.

Read `README.md` for the general workflow and `docs/lean_pipeline.md` before
changing Lean experiments. Executable code and current configs take precedence
over stale examples. Preserve experimental comparability and raw evidence.

## Repository map

| Location | Responsibility |
| --- | --- |
| `main.py` | Hydra entry point for chat, prediction generation, and benchmark dispatch; instantiates model, logger, controller, and selected tools. |
| `benchmark.py` | Standalone benchmark entry point. |
| `src/agent/controller.py` | Tool interaction loop, plain structured output, retries, forced finalization, token accounting. |
| `src/agent/schemas.py` | Pydantic tool arguments and final-answer contracts for plain, Sage, and Lean agents. |
| `src/agent/verification.py`, `controller_utils.py` | Verification and controller helpers. |
| `src/tools/catalog.py` | Tool names and LangChain wrappers; Context7 loading lives in `context7.py`. |
| `src/sage/` | Docker-backed Sage execution and typed execution results. |
| `src/lean/` | Lean REPL/oneshot backends, resource management, declaration and axiom checks. |
| `src/benchmark/generate_predictions.py` | Dataset iteration, retries, per-row prediction logs and summaries. |
| `src/benchmark/benchmark.py`, `components/` | SymPy comparison, judge stage, metadata enrichment, output paths, statistics. |
| `src/benchmark/lean_pipeline.py` | Current protocol: task building, source/label joins, certificate replay, audit templates, analysis. |
| `scripts/run_lean_verification.py` | Dedicated Hydra runner for Lean verification, manifests, resume, concurrency, dry runs. |
| `scripts/build_lean_tasks.py`, `analyze_lean_results.py` | Earlier Lean task/analysis workflows; distinguish these from the current pipeline. |
| `configs/`, `prompts/` | Hydra config groups and prompt/skill text assets. |
| `src/utils/` | Config/text resolution, provider compatibility, structured output and logging adapters. |
| `tests/` | Mostly mocked Python tests, plus real Lean integration tests. |
| `lean_ws/` | Pinned Lean/Mathlib workspace; generated dependencies are local. |

Many files visible in the working directory are local research material, not
portable dependencies. `git ls-files` is the source of truth for tracked content.
`paper/`, `eacl/`, `notes/`, `reports/`, `drafts/`, `outputs/`, `artifacts/`,
`wandb/`, and most data/scripts are ignored. Currently only three scripts are
tracked: `build_lean_tasks.py`, `run_lean_verification.py`, and
`analyze_lean_results.py`. Two tracked datasets are
`data/processed/agent_benchmark_5.json` and
`data/processed/normalized_realmath.json`; do not assume other datasets or saved
results exist in a fresh checkout.

The ignore rules include `scripts/*`, `data/*`, `*.json*`, `*.txt`, and `*.png`.
Check `git check-ignore -v PATH` when a new source or fixture is absent from the
diff. Add a narrow exception for intentionally versioned files rather than
force-adding generated research output. Do not clean ignored directories: they
can contain valuable experiments, manuscripts, credentials, and large installs.

## Environment and setup

Run commands from the repository root. Python must be `>=3.13,<3.14`; use `uv`
and preserve `uv.lock`. Base setup is `uv sync` (includes the default dev group).
Optional extras are `analysis`, `deepseek`, `grok`, `qwen`, and `mlx`.
`uv sync --extra analysis` adds pandas/seaborn for analysis. An existing
`.venv/bin/python` can run local checks without a dependency sync or network.

Use `.env.example` as a list of example variables, not a complete credential
inventory. Never print, copy into documentation, or commit `.env` values.
Make's run targets load `.env` via `uv run --env-file .env`; plain Python
commands should not be assumed to load it. Configure only the selected provider.
In particular, the default prediction and Lean configs use OpenRouter and require
`OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL`. Inspect the chosen
`configs/model/*.yaml` for exact settings. Provider model identifiers in configs
are experiment choices, not a promise of current endpoint availability.

Use `logger=console` for local work. The chat default is W&B, which needs its
account/project configuration and can upload logs. Context7 is optional and uses
`CONTEXT7_API_KEY` and optionally `CONTEXT7_MCP_URL`. Model, judge, documentation,
and logging calls can use external services; do not launch full experiments as a
routine validation step. Use small explicit limits when a live pilot is in scope.

Sage needs Docker running. Set `SAGEMATH_IMAGE` to the intended image, preferably
a digest for reproducibility. The config falls back to `sagemath:latest` and
`SAGEMATH_PLATFORM=linux/amd64`; check architecture compatibility. Preserve the
runtime's network isolation, read-only filesystem, dropped capabilities,
resource limits, and timeout cleanup. Sage runs preparsed code with Sage globals;
the tool normally reads the final value from `RESULT`.

Lean is pinned to `leanprover/lean4:v4.34.0-rc1`, with Mathlib revision
`v4.34.0-rc1` in `lean_ws/lakefile.toml`. The default runtime expects
`tools/lean-repl/.lake/build/bin/repl` and a populated `lean_ws/.lake/`.
These installations are ignored and are not supplied by `uv sync`. Inspect their
availability and version compatibility before integration tests; do not silently
upgrade pins. `LEAN_BACKEND` selects `repl` or `oneshot`; `LEAN_POOL_SIZE`
defaults to 1. A Mathlib worker can use roughly 6 GB RAM, so increase pool size
only deliberately. Default startup and call timeouts are 180 and 60 seconds.

## Commands and known configuration traps

Inspect composed configs without constructing models or running tools:

```bash
.venv/bin/python main.py system_prompt=tool logger=console --cfg job
.venv/bin/python main.py --config-name generate_predictions --cfg job
.venv/bin/python scripts/run_lean_verification.py --config-name lean_pipeline --cfg job
```

Do not use `--resolve` when displaying configs if that could expose credentials.
Hydra overrides for controller settings belong under `controller.config`, e.g.
`controller.config.max_steps=12`. `configs/chat.yaml` currently also contains a
literal root key `controller.max_steps: 1`; it does not override that nested value.

Known stale references at the time this guide was written:

- Chat defaults to `system_prompt: v3`, but only `tool`, `no-tool`, and `lean`
  config files exist. Supply `system_prompt=tool` for Sage chat.
- `make generate-predictions-tool` selects nonexistent `system_prompt=v4`.
  Prefer the direct prediction command below.
- Some chat prompt configs reference ignored `.txt` files. Use an explicit
  `prompt.text` and `prompt.file=null` for a portable smoke run.
- Make's `benchmark` and `generate-predictions` targets use macOS `caffeinate`.
  Direct Python commands work without that wrapper.

Examples below perform live model calls; adjust the model and input deliberately:

```bash
# Sage chat with a self-contained prompt
uv run --env-file .env python main.py system_prompt=tool logger=console \
  'prompt.text=Compute 2 + 2.' prompt.file=null

# Plain chat
uv run --env-file .env python main.py system_prompt=no-tool 'tools=[]' \
  logger=console 'prompt.text=Compute 2 + 2.' prompt.file=null

# One prediction, using an explicitly supplied dataset
uv run --env-file .env python main.py --config-name generate_predictions \
  tool_mode=tool 'tools=[sage_exec]' \
  generate_predictions.config.dataset_path=data/processed/agent_benchmark_5.json \
  generate_predictions.config.limit=1

# Evaluation of existing predictions (includes configured external judges)
uv run --env-file .env python benchmark.py \
  benchmark.config.predictions_path=outputs/generate_predictions/REPLACE_ME.jsonl
```

Prediction `tool_mode=tool` normally enables both `sage_exec` and Context7
`query-docs`; the example limits it to Sage. Use `tool_mode=no_tool` for the
configured no-tool arm. Tool modes also set model options such as
`use_responses_api`; check provider compatibility when switching model groups.

## Data and output contracts

Prediction input fields default to `id`, `question`, `answer`, and `sympy_answer`.
The runner writes uniquely named `predictions_*.jsonl` and summary files under
`outputs/generate_predictions/`, recording answers, explanation, confidence,
traces, stop reason, token usage, elapsed time, and errors. Preserve problem IDs,
raw model answers, normalized answers, and trace/error metadata when transforming
results; downstream comparisons and Lean label joins depend on them.

Final-answer schemas reject extra fields. `sympy_answer` is a nonempty string or
list of nonempty strings without LaTeX wrappers/backslashes; confidence is 1–5.
Lean final answers extend the Sage schema so controller finalization paths remain
compatible. Update schemas, prompts, consumers, and relevant tests together when
changing this contract. Preserve distinctions between runtime success, verified
claims, model verdicts, and scored correctness.

## Lean experiment protocol (critical)

Read the complete commands and rationale in `docs/lean_pipeline.md`. The current
protocol is `sage-lean-v1`, selected by `--config-name lean_pipeline`. `main.py`
does not dispatch Lean verification: use `scripts/run_lean_verification.py`.

A minimal preparation workflow (requires the specified saved results locally):

```bash
.venv/bin/python -m src.benchmark.lean_pipeline build \
  --results data/results/all_results_current.json --limit 8 \
  --out data/processed/lean_pipeline_pilot8.json

.venv/bin/python scripts/run_lean_verification.py --config-name lean_pipeline \
  lean_verification.tasks_path=data/processed/lean_pipeline_pilot8.json \
  lean_verification.run_name=pilot8_explanation lean_verification.dry_run=true
```

The dry run writes exact prompts and a manifest without starting Lean or calling
models. For an authorized live run, use the same arguments with `dry_run=false`
and load provider environment variables. Use `audit-template` and `analyze`
subcommands documented in `docs/lean_pipeline.md` afterward.

Maintain these scientific and implementation invariants:

- Build from saved tool-arm answers across answer types. Exclude empty answers
  and source errors with counts. Selection is seeded and independent of labels;
  do not balance labels or add formalizability filtering to this protocol.
- Use the exact question seen by the Sage agent, not a source theorem containing
  the reference answer. Keep original/normalized answers and failed tool traces.
- Label joins must match problem ID, question, candidate, and original answer.
  Missing/ambiguous labels remain unknown. Judge labels are not mathematical truth
  and must not leak into agent prompts or blinded faithfulness review templates.
- Use the same saved task file for `answer`, `explanation`, and `traces` context
  ablations. Do not silently truncate evidence or feed Lean feedback into Sage.
- Keep manifests and original JSONL together. Manifests bind tasks, prompts,
  config, sources, lockfile, and Lean versions. Resume skips completed attempts
  and retries exceptions; reject changed manifests or malformed results. Choose
  a new run name after code/protocol/config changes. Never concatenate runs with
  different manifests or pool legacy `trackB_mini` with this protocol.
- Final code must define `candidate_claim : Prop` and root declaration
  `candidate_certificate : candidate_claim` (PROVED) or
  `candidate_certificate : ¬ candidate_claim` (REFUTED). Replay the final snippet
  in a fresh base environment with a typed wrapper and axiom inspection. An
  intermediate successful tool call, failed `decide`, or verbal counterexample
  does not establish a final certificate.
- Lean tool calls are stateless with Mathlib preloaded; reject snippet imports.
  `status == "ok"` alone is not proof: `sorry` can elaborate successfully.
  Preserve declaration checks and the axiom allowlist (`propext`,
  `Classical.choice`, `Quot.sound`); reject invented axioms and keep
  `native_decide` banned by default.
- A kernel certificate does not certify faithful natural-language translation.
  Separate claimed verdicts, `kernel_proved`/`kernel_refuted`, and
  `audited_proved`/`audited_refuted`. Faithfulness reviews must bind to exact
  task/code hashes and examine definitions, quantifiers, assumptions, non-vacuity,
  and completeness of the requested conclusion.
- UNKNOWN is abstention. Include pending/errors and the full selected-task
  denominator in reports. Answers from multiple models on the same problem are
  dependent: report unique problems and use paired/problem-clustered uncertainty
  for comparisons rather than naive pooled binomial confidence intervals.

Default Lean runs use four agent threads sharing one runtime, 12 controller
steps, and two finalization retries. Final certificate replay is an additional
Lean call outside the agent tool budget. Keep these costs explicit in comparisons.

## Validation and change discipline

Use focused tests first, then broaden based on the change. Useful commands:

```bash
# Python suite excluding the real-Lean integration module
.venv/bin/python -m pytest --ignore=tests/test_lean_runtime.py -q

# Dataset/protocol changes
.venv/bin/python -m pytest tests/test_lean_pipeline.py tests/test_build_lean_tasks.py -q

# Real Lean certificate checks (requires the local Lean installation)
.venv/bin/python -m pytest tests/test_lean_runtime.py -k pipeline_final_certificate -q

# Full suites, including real Lean tests
make test
make test-analysis
```

Sage runtime tests mock Docker calls. Lean runtime tests actually start Lean;
there is no automatic missing-install skip. A full test run is not purely mocked.
No formatter/linter or CI workflow is configured in the tracked project at the
time of writing; follow surrounding Python style and avoid unrelated formatting.
For documentation-only changes, check paths, commands/config composition, and the
diff rather than launching paid runs or expensive integration tests.

When editing, inspect existing tests for the subsystem: controller/schema changes
in `test_controller.py`; tool wiring in `test_tool_catalog.py`/`test_context7.py`;
logging in `test_experiment_logger.py`; benchmark and normalization behavior in
`test_benchmark.py`, `test_compare_predictions.py`, `test_sympy_comparison.py`;
prediction persistence in `test_generate_predictions.py`. Add regression coverage
for changed behavior, especially certificate acceptance and evidence integrity.

Before handing off, review `git diff --check`, changed files, and working-tree
status. Report what changed, checks actually run, and remaining prerequisites or
failures. Do not claim a live benchmark or proof verification succeeded based
only on mocked tests or a config dry run. Keep this guide updated when entry
points, contracts, pins, or known configuration issues change.
