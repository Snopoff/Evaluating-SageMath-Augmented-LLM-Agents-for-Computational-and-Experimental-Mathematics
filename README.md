# LLMxM2

Minimal LLM + Sage experimentation codebase.

## Design

The main path is intentionally small:

- `main.py`
- `src/agent/`
- `src/tools/`
- `src/sage/`
- `src/benchmark/`

No MCP server layer in the main run path and no large policy framework.

## Quickstart

1. Install dependencies:

```bash
uv sync
```

2. Install Docker and make sure the Docker daemon is running.

The Sage runtime executes via `docker run`, so Docker is a required local dependency.

3. Set environment variables:

```bash
export OPENAI_API_KEY=...
export SAGEMATH_IMAGE='docker.io/sagemath/sagemath@sha256:<real_digest>'
```

`SAGEMATH_IMAGE` must point to a real Sage image digest. The default config value is only a placeholder.

Optional (Apple Silicon with amd64 image):

```bash
export SAGEMATH_PLATFORM='linux/amd64'
```

Optional: pre-pull the image once before running. If the image is missing locally, `docker run` will usually pull it automatically.

```bash
docker pull "$SAGEMATH_IMAGE"
```

## Run

Chat:

```bash
uv run --env-file .env python main.py mode=chat model=openai
```

Benchmark:

```bash
uv run --env-file .env python main.py mode=benchmark benchmark.limit=25
```

## Hydra Construction

Models, the Sage runtime, and the controller are instantiated directly through Hydra in `main.py`:

```python
model = instantiate(cfg.model)
runtime = instantiate(cfg.sage, logger=logger)
controller = instantiate(cfg.controller, model=model, tools=tools, logger=logger)
```

Model profiles live in `configs/model/*.yaml` and use each LangChain provider class as `_target_`.
The Sage runtime profile lives in `configs/sage/default.yaml`.

## Tool Extension Point

Tools are selected by Hydra name in `configs/chat.yaml` and `configs/benchmark.yaml`:

```yaml
tools:
  - sage_exec
```

`src/tools/catalog.py` currently exposes one minimal built-in:

- `sage_exec`: run raw Sage code in Docker (`code`, optional `result_var`, optional `timeout_sec`)
- `submit_final_answer`: internal finalization tool added by the controller

To add a new tool:

1. Add a Pydantic args schema in `src/agent/schemas.py` if the tool needs arguments.
2. Add a LangChain `@tool` factory in `src/tools/catalog.py`.
3. Add the factory to `AVAILABLE_TOOLS`.
4. Add the tool name to `tools` in the Hydra config.

## Benchmark Outputs

Benchmark mode writes:

- `predictions.jsonl`
- `tool_traces.jsonl`
- `metrics.json`
