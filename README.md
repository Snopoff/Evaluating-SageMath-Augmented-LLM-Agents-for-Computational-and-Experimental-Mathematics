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
