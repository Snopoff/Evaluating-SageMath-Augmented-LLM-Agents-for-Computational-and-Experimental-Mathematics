# Evaluating SageMath-Augmented LLM Agents for Computational and Experimental Mathematics

This repository contains the code for running LLM agents that solve mathematical
problems with optional SageMath tool use. The implementation is intentionally
small: Hydra configs select the model, prompt, logger, controller, and tools;
`AgentController` drives the interaction loop; and Sage code is executed in a
Docker-backed runtime.

## Repository Layout

- `main.py`: Hydra entry point for chat, prediction generation, and benchmark modes.
- `benchmark.py`: standalone Hydra entry point for benchmark evaluation.
- `configs/`: model, controller, prompt, logger, Sage, Context7, and benchmark configs.
- `src/agent/`: agent controller, schemas, and answer verification helpers.
- `src/sage/`: Docker-backed Sage runtime.
- `src/tools/`: LangChain tool wrappers for Sage and Context7.
- `src/benchmark/`: prediction, comparison, judging, and benchmark utilities.
- `prompts/`: reusable system prompts and Sage usage notes.
- `tests/`: unit tests.

## Installation

Install the base dependencies with `uv`:

```bash
uv sync
```

Optional provider integrations are installed as extras:

```bash
uv sync --extra deepseek
uv sync --extra grok
uv sync --extra qwen
```

For analysis-only dependencies, use:

```bash
uv sync --extra analysis
```

## Environment

Sage execution uses Docker, so Docker must be installed and running. Set a Sage
image before using `sage_exec`:

```bash
export SAGEMATH_IMAGE='docker.io/sagemath/sagemath@sha256:<real_digest>'
```

If the image is built for `linux/amd64` and the host is Apple Silicon, also set:

```bash
export SAGEMATH_PLATFORM='linux/amd64'
```

Set only the model-provider keys needed for the run:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export DEEPSEEK_API_KEY=...
export XAI_API_KEY=...
export QWEN_API_KEY=...
```

Yandex-compatible runs additionally use:

```bash
export YANDEX_API_KEY=...
export YANDEX_FOLDER_ID=...
```

The default chat config uses the W&B logger. Either configure W&B:

```bash
export WANDB_ENTITY=...
export WANDB_PROJECT=...
```

or override the logger on the command line:

```bash
make chat HYDRA='logger=console'
```

Context7 documentation tools are optional:

```bash
export CONTEXT7_API_KEY=...
export CONTEXT7_MCP_URL='https://mcp.context7.com/mcp'
```

## Running

Run the default chat configuration:

```bash
make chat
```

Override model/provider settings through Hydra:

```bash
make chat HYDRA='model=openai model_name=gpt-5.5 logger=console'
make chat HYDRA='model=deepseek model_name=deepseek-chat logger=console'
```

Disable tools for a plain structured LLM run:

```bash
make chat-no-tool HYDRA='model=openai model_name=gpt-5.5 logger=console'
```

Enable Context7 together with Sage:

```bash
make chat HYDRA='model=openai model_name=gpt-5.5 logger=console "tools=[sage_exec,query-docs]"'
make chat HYDRA='model=openai model_name=gpt-5.5 logger=console "tools=[sage_exec,resolve-library-id,query-docs]"'
```

Generate predictions with the `generate_predictions` config:

```bash
make generate-predictions
make generate-predictions-no-tool HYDRA='model=openai model_name=gpt-5.5 logger=console'
make generate-predictions-tool HYDRA='model=google model_name=gemini-3.1-pro-preview logger=console'
```

Run benchmark evaluation against an existing predictions file:

```bash
make benchmark HYDRA='benchmark.config.predictions_path=data/results/<path-to-predictions>.json'
```

Judge a SymPy comparison output file:

```bash
make judge-sympy input=data/results/<model>/<setup>/output/<file>.json
```

Run tests:

```bash
make test
make test-analysis
```

## Notes

- `tools=[sage_exec]` runs the Sage-backed ReAct loop.
- `tools=[]` runs the same controller without Sage.
- `tools=[sage_exec,query-docs]` adds Context7 documentation lookup.
- Runtime artifacts, logs, W&B files, generated result files, and `.env` are ignored by default.
