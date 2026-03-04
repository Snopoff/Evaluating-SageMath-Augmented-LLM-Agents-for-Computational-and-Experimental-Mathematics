# LLMxM2

Fail-closed SageMath MCP stack for OpenAI-compatible models (OpenAI, Ollama, MLX endpoints) with RealMath benchmarking.

## Quickstart

1. Install dependencies:

```bash
uv sync
```

2. Set API keys / runtime image in `.env` or shell:

```bash
export OPENAI_API_KEY=...
export SAGEMATH_IMAGE='docker.io/sagemath/sagemath@sha256:<real_digest>'
```

Apple Silicon (`arm64`) note: if your Sage image is only available for `amd64`, set:

```bash
export SAGEMATH_PLATFORM='linux/amd64'
```

If Dockerized Sage fails with command-not-found or return code `127`, first use the image default user:

```bash
unset SAGEMATH_USER
```

Only if needed for debugging, temporarily run as root:

```bash
export SAGEMATH_USER='0:0'
```

If needed, force shell-based entrypoint resolution:

```bash
export SAGEMATH_ENTRYPOINT='/bin/bash'
```

Sage startup under `--read-only` requires writable HOME and `.sage`; defaults are already set to `/tmp` via MCP config (`HOME=/tmp`, `DOT_SAGE=/tmp/.sage`).

## Modes

### Chat

Run controller-managed solving with optional tool use:

```bash
uv run --env-file .env python main.py mode=chat provider=openai model=openai
```

For prompts with many symbols, use a prompt file:

```bash
uv run --env-file .env python main.py mode=chat provider=ollama model=qwen3_4b_thinking \
  prompt_file=/absolute/path/to/question.txt
```

To require at least one tool call before finalization:

```bash
uv run --env-file .env python main.py mode=chat provider=openai model=openai \
  controller.tool_use_mode=required controller.min_required_tool_calls=1
```

### Benchmark

```bash
uv run --env-file .env python main.py mode=benchmark benchmark.limit=25
```

### Serve MCP

```bash
uv run --env-file .env python main.py mode=serve_mcp
```

Open MCP inspector locally:

```bash
uv run --env-file .env mcp dev src/llmxm2/mcp/dev_app.py:mcp
```

## Tool API

Tool name: `sage_eval`

Common payload shape:

```json
{
  "operation": "factor",
  "args": {
    "positional_args": ["x^2 - 1"],
    "keyword_args": {},
    "coerce_symbolic_strings": true
  },
  "assumptions": {"domain": "QQ"},
  "request_id": "example-1",
  "budget_profile": "conservative"
}
```

### Arbitrary Sage callable execution

You can call any callable available in the Sage runtime namespace by setting `operation` to the callable name and passing `args.positional_args` / `args.keyword_args`.

Example:

```json
{
  "operation": "PolynomialRing",
  "args": {
    "positional_args": ["QQ", ["x", "y"]],
    "keyword_args": {"order": "lex"},
    "coerce_symbolic_strings": true
  },
  "assumptions": {"domain": "QQ"},
  "request_id": "poly-ring-1",
  "budget_profile": "conservative"
}
```

### `sage_snippet`

Use `sage_snippet` for multi-step or custom workflows:

```json
{
  "operation": "sage_snippet",
  "args": {
    "code": "from sage.all import *\nR = PolynomialRing(QQ, ['x','y'])\nx, y = R.gens()\nRESULT = factor(x**2-y**2)",
    "result_var": "RESULT",
    "include_locals": false
  },
  "assumptions": {"domain": "ZZ"},
  "request_id": "snippet-demo-1",
  "budget_profile": "conservative"
}
```

## Prompt assets

Reusable prompt templates are under:

- `/Users/snopoff/Documents/Research/LLMxM2/prompts/frontiermath_degree19_blind.txt`
- `/Users/snopoff/Documents/Research/LLMxM2/prompts/frontiermath_degree19_dickson.txt`
- `/Users/snopoff/Documents/Research/LLMxM2/prompts/join_chromatic_pattern.txt`
- `/Users/snopoff/Documents/Research/LLMxM2/prompts/prompt_sagemath_signed_conjecture.md`

## Security defaults

- Docker execution: network-disabled, read-only filesystem, dropped caps, no-new-privileges.
- Static fail-closed policy gates payload size/safety before execution.
- Per-call and cumulative budgets are enforced by controller + executor configs.

## Outputs

In benchmark mode, files are written to the Hydra run directory (for example under `outputs/...`):

- `predictions.jsonl`
- `tool_traces.jsonl`
- `metrics.json`
