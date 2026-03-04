import json
import logging
import rootutils
from pathlib import Path

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.llmxm2.agent.controller import AgentController, ControllerConfig
from src.llmxm2.benchmark.run_realmath import BenchmarkConfig, RealMathBenchmarkRunner
from src.llmxm2.mcp.client import InProcessSageToolClient
from src.llmxm2.mcp.sage_server import SageMCPService, run_mcp_server


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

    #! `from_config` -> `hu.instantiate`
    # * ======
    mcp_cfg = OmegaConf.to_container(cfg.mcp, resolve=True)
    if not isinstance(mcp_cfg, dict):
        raise ValueError("Config 'mcp' must be a mapping.")

    mcp_cfg.setdefault("progress_logs", progress_logs)
    _progress(progress_logs, "initializing Sage MCP service")
    service = SageMCPService.from_config(mcp_cfg)
    _progress(progress_logs, "Sage MCP service ready")
    # * ======

    if mode == "serve_mcp":
        transport = str(cfg.mcp.server.transport)
        _progress(progress_logs, f"serving MCP over transport={transport}")
        run_mcp_server(service, transport=transport)
        return

    _progress(progress_logs, "initializing model client")
    client = instantiate(cfg.provider.client)

    #! `from_config` -> `hu.instantiate`
    # * ======
    controller_cfg = ControllerConfig.from_config(OmegaConf.to_container(cfg.controller, resolve=True))
    tool_client = InProcessSageToolClient(service=service)

    controller = AgentController(
        client=client,
        model_name=str(cfg.model.name),
        tool_client=tool_client,
        config=controller_cfg,
    )
    _progress(progress_logs, f"controller ready (model={cfg.model.name})")
    # * ======

    if mode == "chat":
        prompt = str(cfg.get("prompt", ""))
        prompt_file = cfg.get("prompt_file")
        if prompt_file is not None and str(prompt_file):
            prompt_path = Path(to_absolute_path(str(prompt_file)))
            _progress(progress_logs, f"loading prompt from file: {prompt_path}")
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        _progress(progress_logs, f"starting chat solve (prompt_chars={len(prompt)})")
        result = controller.solve(prompt)
        _progress(
            progress_logs,
            f"chat solve completed (stop_reason={result.stop_reason}, turns={result.turn_count})",
        )
        print(result.final_answer)
        return

    if mode == "benchmark":
        bench_cfg_raw = OmegaConf.to_container(cfg.benchmark, resolve=True)
        if not isinstance(bench_cfg_raw, dict):
            raise ValueError("Config 'benchmark' must be a mapping.")

        dataset_path = Path(to_absolute_path(str(cfg.benchmark.dataset_path)))
        benchmark_cfg = BenchmarkConfig.from_config(bench_cfg_raw, dataset_path=dataset_path)
        _progress(
            progress_logs,
            f"starting benchmark (dataset={dataset_path}, limit={benchmark_cfg.limit})",
        )
        runner = RealMathBenchmarkRunner(controller=controller, tool_client=tool_client, config=benchmark_cfg)
        metrics = runner.run()
        _progress(progress_logs, "benchmark completed")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    raise ValueError(f"Unsupported mode: {mode!r}")


if __name__ == "__main__":
    main()
