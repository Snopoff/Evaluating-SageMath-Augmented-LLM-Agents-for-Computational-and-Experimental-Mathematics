import json
import logging
from pathlib import Path
from typing import Any

import hydra
import rootutils
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.agent.controller import AgentController, ControllerConfig
from src.benchmark.run_realmath import BenchmarkConfig, RealMathBenchmarkRunner
from src.sage.runtime import SageRuntime
from src.tools.registry import ToolRegistry
from src.tools.types import ToolResult


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[progress] {message}", flush=True)


def _build_tool_registry(runtime: SageRuntime) -> ToolRegistry:
    registry = ToolRegistry()

    def sage_exec(arguments: dict[str, Any]) -> ToolResult:
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            return ToolResult(ok=False, content="sage_exec requires 'code' as a non-empty string")

        result_var = arguments.get("result_var", "RESULT")
        if not isinstance(result_var, str) or not result_var.strip():
            result_var = "RESULT"

        timeout = arguments.get("timeout_sec")
        timeout_sec: float | None = float(timeout) if isinstance(timeout, (int, float)) else None

        result = runtime.execute_sage_code(
            code=code,
            result_var=result_var,
            timeout_sec=timeout_sec,
        )

        content = result.result_plain
        if not content and result.stdout.strip():
            content = result.stdout.strip()
        if result.status != "ok":
            content = result.error or result.stderr.strip() or "Sage execution failed"

        return ToolResult(
            ok=result.status == "ok",
            content=content,
            metadata={
                "status": result.status,
                "runtime_ms": result.runtime_ms,
                "stderr": result.stderr,
                "result_latex": result.result_latex,
            },
        )

    registry.register(
        name="sage_exec",
        schema={
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "result_var": {"type": "string"},
                "timeout_sec": {"type": "number"},
            },
            "required": ["code"],
        },
        handler=sage_exec,
        description="Execute raw Sage code inside Docker.",
    )

    return registry


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    _setup_logging()

    mode = str(cfg.get("mode", "chat"))
    progress_logs = bool(cfg.get("progress_logs", True))
    _progress(progress_logs, f"starting main (mode={mode})")

    # Keep provider construction Hydra-driven.
    client = instantiate(cfg.provider.client)

    sage_cfg = OmegaConf.to_container(cfg.sage, resolve=True)
    if not isinstance(sage_cfg, dict):
        raise ValueError("Config 'sage' must be a mapping.")

    sage_cfg.setdefault("progress_logs", progress_logs)
    runtime = SageRuntime.from_config(sage_cfg)
    tools = _build_tool_registry(runtime)

    controller_cfg = OmegaConf.to_container(cfg.controller, resolve=True)
    if not isinstance(controller_cfg, dict):
        raise ValueError("Config 'controller' must be a mapping.")

    controller = AgentController(
        client=client,
        model_name=str(cfg.model.name),
        tool_registry=tools,
        config=ControllerConfig.from_config(controller_cfg),
    )

    if mode == "chat":
        prompt = str(cfg.get("prompt", ""))
        prompt_file = cfg.get("prompt_file")
        if prompt_file is not None and str(prompt_file):
            prompt_path = Path(to_absolute_path(str(prompt_file)))
            _progress(progress_logs, f"loading prompt from file: {prompt_path}")
            prompt = prompt_path.read_text(encoding="utf-8").strip()

        result = controller.solve(prompt)
        _progress(progress_logs, f"chat completed (turns={result.turn_count}, reason={result.stop_reason})")
        print(result.final_answer)
        return

    if mode == "benchmark":
        bench_cfg = OmegaConf.to_container(cfg.benchmark, resolve=True)
        if not isinstance(bench_cfg, dict):
            raise ValueError("Config 'benchmark' must be a mapping.")

        dataset_path = Path(to_absolute_path(str(cfg.benchmark.dataset_path)))
        benchmark_cfg = BenchmarkConfig.from_config(bench_cfg, dataset_path=dataset_path)
        runner = RealMathBenchmarkRunner(controller=controller, config=benchmark_cfg, sage_runtime=runtime)
        metrics = runner.run()
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    raise ValueError(f"Unsupported mode: {mode!r}. Use 'chat' or 'benchmark'.")


if __name__ == "__main__":
    main()
