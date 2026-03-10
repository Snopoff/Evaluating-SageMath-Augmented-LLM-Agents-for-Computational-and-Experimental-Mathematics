from datetime import datetime
from pathlib import Path

import hydra
import hydra.utils as hu
import rootutils
from omegaconf import DictConfig, OmegaConf

from typing import Iterable

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.agent.controller import AgentController, ControllerConfig  # noqa: E402
from src.sage.runtime import SageRuntime  # noqa: E402
from src.tools.catalog import AVAILABLE_TOOLS  # noqa: E402
from src.tools.registry import ToolRegistry  # noqa: E402
from src.utils.logging import setup_logging, progress  # noqa: E402
from src.utils.config_helpers import resolve_prompt  # noqa: E402


def _save_verified_sage_code(code: str) -> Path:
    artifact_dir = Path(__file__).resolve().parent / "artifacts" / "verified_sage_code"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    artifact_path = artifact_dir / f"verified_{timestamp}.py"
    artifact_path.write_text(code, encoding="utf-8")
    return artifact_path


@hydra.main(version_base=None, config_path="configs", config_name="chat")
def main(cfg: DictConfig) -> None:
    setup_logging()

    mode = str(cfg.get("mode", "chat"))
    progress_logs = bool(cfg.get("progress_logs", True))
    if progress_logs:
        progress(f"starting main (mode={mode})")

    client = hu.instantiate(cfg.provider.client)

    sage_cfg = OmegaConf.to_container(cfg.sage, resolve=True)
    if not isinstance(sage_cfg, dict):
        raise ValueError("Config 'sage' must be a mapping.")
    runtime = SageRuntime.from_config(sage_cfg)  # type: ignore

    tool_names = cfg.get("tools", [])
    if not isinstance(tool_names, Iterable) or not all(isinstance(name, str) for name in tool_names):
        raise ValueError("Config 'tools' must be a list of tool names.")

    tools = ToolRegistry()
    available_names = sorted(AVAILABLE_TOOLS)
    for tool_name in tool_names:
        factory = AVAILABLE_TOOLS.get(tool_name)
        if factory is None:
            available_text = ", ".join(available_names) if available_names else "(none)"
            raise ValueError(f"Unknown tool: {tool_name!r}. Available tools: {available_text}")
        tools.register(factory(runtime))

    if progress_logs:
        progress(f"initialized tools: [bold orange1]{', '.join(tool.name for tool in tools.list_tools())}[/bold orange1]")

    controller_cfg = OmegaConf.to_container(cfg.controller, resolve=True)
    if not isinstance(controller_cfg, dict):
        raise ValueError("Config 'controller' must be a mapping.")

    controller = AgentController(
        client=client,
        model_name=str(cfg.model.name),
        tool_registry=tools,
        config=ControllerConfig.from_config(controller_cfg),  # type: ignore
    )

    if mode == "chat":
        prompt = resolve_prompt(cfg.prompt, progress_logs)

        result = controller.solve(prompt)
        artifact_path: Path | None = None
        if result.verified_sage_code.strip():
            artifact_path = _save_verified_sage_code(result.verified_sage_code)
        if progress_logs:
            progress(f"chat completed (turns={result.turn_count}, reason={result.stop_reason})")
        print(result.final_answer)
        if artifact_path is not None:
            print()
            print(f"Verified Sage code saved to: {artifact_path}")
            print(result.verified_sage_code)
        return

    if mode == "benchmark":
        bench_cfg = OmegaConf.to_container(cfg.benchmark, resolve=True)
        if not isinstance(bench_cfg, dict):
            raise ValueError("Config 'benchmark' must be a mapping.")

        dataset_path = Path(hu.to_absolute_path(str(cfg.benchmark.dataset_path)))
        benchmark_cfg = hu.instantiate(cfg.benchmark, dataset_path=dataset_path)
        # runner = hu.instantiate(cfg.benchmark.runner, controller=controller, config=benchmark_cfg, sage_runtime=runtime)
        # metrics = runner.run()
        # print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return

    raise ValueError(f"Unsupported mode: {mode!r}. Use 'chat' or 'benchmark'.")


if __name__ == "__main__":
    main()
