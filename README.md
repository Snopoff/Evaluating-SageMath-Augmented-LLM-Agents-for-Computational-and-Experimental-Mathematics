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

In `configs/chat.yaml` select the proper model, prompt and other parameters, and run

```Makefile
make run
```

Equivalently, you can pass the arguments right in CLI

```Makefile
make run model=openai
```

Use the same base configs for both Sage-backed and plain runs. For the plain structured
LLM variants, override only the tool list and, optionally, the system prompt:

```bash
uv run --env-file .env python main.py --config-name chat model=openai model_name=gpt-5.5 system_prompt=no-tool 'tools=[]'
uv run --env-file .env python main.py --config-name benchmark system_prompt=no-tool 'tools=[]' benchmark.config.output_dir=outputs/agent_benchmark_5_plain
```

The same `AgentController` is used for plain and Sage-backed runs. With `tools: []`,
it makes one structured model call. With `tools: [sage_exec]`, it runs the Sage
ReAct loop and finalizes with the Sage
structured schema.
