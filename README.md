# LLMxM2

Minimal LLM + Sage experimentation codebase.

## Design

The main path is intentionally small:

- `main.py`
- `src/llmxm2/agent/`
- `src/llmxm2/tools/`
- `src/llmxm2/sage/`
- `src/llmxm2/benchmark/`

No MCP server layer in the main run path and no large policy framework.

## Quickstart

1. Install dependencies:

```bash
uv sync
```

2. Set environment variables:

```bash
export OPENAI_API_KEY=...
export SAGEMATH_IMAGE='docker.io/sagemath/sagemath@sha256:<real_digest>'
```

Optional (Apple Silicon with amd64 image):

```bash
export SAGEMATH_PLATFORM='linux/amd64'
```

## Run

Chat:

```bash
uv run --env-file .env python main.py mode=chat provider=openai model=openai
```

Benchmark:

```bash
uv run --env-file .env python main.py mode=benchmark benchmark.limit=25
```

## Hydra Provider Construction

Provider clients are instantiated directly through Hydra in `main.py`:

```python
client = instantiate(cfg.provider.client)
```

Model selection stays in Hydra config (`configs/model/*.yaml`).

## Tool Extension Point

Tools are selected by Hydra name in `configs/default.yaml`:

```yaml
tools:
  enabled:
    - sage_exec
```

`src/tools/catalog.py` currently exposes one minimal built-in:

- `sage_exec`: run raw Sage code in Docker (`code`, optional `result_var`, optional `timeout_sec`)

To add a new tool:

1. Add a plain handler and `make_<tool>_tool(...)` factory in `src/tools/catalog.py`.
2. Add the factory to `AVAILABLE_TOOLS`.
3. Add the tool name to `tools.enabled`.

## Benchmark Outputs

Benchmark mode writes:

- `predictions.jsonl`
- `tool_traces.jsonl`
- `metrics.json`
