import json
import logging
from pathlib import Path

import hydra
import rootutils
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.agent.controller import AgentController, ControllerConfig
from src.benchmark.run_realmath import BenchmarkConfig, RealMathBenchmarkRunner
from src.sage.runtime import SageRuntime
from src.tools.catalog import AVAILABLE_TOOLS
from src.tools.registry import ToolRegistry


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[progress] {message}", flush=True)


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

    tools_cfg = OmegaConf.to_container(cfg.tools, resolve=True)
    if not isinstance(tools_cfg, dict):
        raise ValueError("Config 'tools' must be a mapping.")

    enabled_tools = tools_cfg.get("enabled", [])
    if not isinstance(enabled_tools, list) or not all(isinstance(name, str) for name in enabled_tools):
        raise ValueError("Config 'tools.enabled' must be a list of tool names.")

    tools = ToolRegistry()
    available_names = sorted(AVAILABLE_TOOLS)
    for tool_name in enabled_tools:
        factory = AVAILABLE_TOOLS.get(tool_name)
        if factory is None:
            available_text = ", ".join(available_names) if available_names else "(none)"
            raise ValueError(f"Unknown tool: {tool_name!r}. Available tools: {available_text}")
        tools.register(factory(runtime))

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
