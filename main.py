import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import hydra
import hydra.utils as hu
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)

from src.tools.catalog import AVAILABLE_TOOLS  # noqa: E402
from src.utils.config_helpers import resolve_prompt, resolve_text_asset  # noqa: E402


def _save_verified_sage_code(code: str) -> Path:
    artifact_dir = Path(__file__).resolve().parent / "artifacts" / "verified_sage_code"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    artifact_path = artifact_dir / f"verified_{timestamp}.py"
    artifact_path.write_text(code, encoding="utf-8")
    return artifact_path


@hydra.main(version_base=None, config_path="configs", config_name="chat")
def main(cfg: DictConfig) -> None:
    mode = str(cfg.get("mode", "chat"))
    progress_logs = bool(cfg.get("progress_logs", True))
    logger = hu.instantiate(cfg.logger, mode=mode)
    logger.setup_logging()
    if progress_logs:
        logger.progress(f"starting main (mode={mode})")

    model = hu.instantiate(cfg.model)

    runtime = hu.instantiate(cfg.sage, logger=logger)

    tool_names = cfg.get("tools", [])
    if not isinstance(tool_names, Iterable) or not all(isinstance(name, str) for name in tool_names):
        raise ValueError("Config 'tools' must be a list of tool names.")

    sage_usage_notes = ""
    if "sage_exec" in tool_names and cfg.get("sage_skill") is not None:
        sage_usage_notes = resolve_text_asset(cfg.sage_skill, label="sage_skill", logger=logger)

    system_prompt = ""
    if cfg.get("system_prompt") is not None:
        system_prompt = resolve_text_asset(cfg.system_prompt, label="system_prompt", logger=logger)

    tools = []
    available_names = sorted(AVAILABLE_TOOLS)
    for tool_name in tool_names:
        factory = AVAILABLE_TOOLS.get(tool_name)
        if factory is None:
            available_text = ", ".join(available_names) if available_names else "(none)"
            raise ValueError(f"Unknown tool: {tool_name!r}. Available tools: {available_text}")
        usage_notes = sage_usage_notes if tool_name == "sage_exec" else ""
        tools.append(factory(runtime, usage_notes))

    if progress_logs:
        logger.progress(f"initialized tools: [bold orange1]{', '.join(tool.name for tool in tools)}[/bold orange1]")

    controller = hu.instantiate(cfg.controller, model=model, tools=tools, logger=logger, system_prompt=system_prompt)

    if mode == "chat":
        prompt = resolve_prompt(cfg.prompt, logger=logger)

        result = None
        try:
            result = controller.solve(prompt)
            artifact_path: Path | None = None
            if result.verified_sage_code.strip():
                artifact_path = _save_verified_sage_code(result.verified_sage_code)
                logger.log_artifact(
                    name="verified_sage_code",
                    path=artifact_path,
                    artifact_type="sage-code",
                    metadata={"mode": mode, "verified": True, "agent_id": controller.agent_id},
                )
            if progress_logs:
                logger.progress(f"chat completed (turns={result.turn_count}, reason={result.stop_reason})")
            if result.final_payload:
                print(json.dumps(result.final_payload, indent=2, ensure_ascii=False))
            else:
                print(result.final_answer)
            if artifact_path is not None:
                print()
                print(f"Verified Sage code saved to: {artifact_path}")
                print(result.verified_sage_code)
            logger.finish_run(status=result.stop_reason)
            return
        except Exception:
            logger.finish_run(status="failed")
            raise

    if mode == "benchmark":
        cfg.benchmark.config.dataset_path = hu.to_absolute_path(str(cfg.benchmark.config.dataset_path))
        runner = hu.instantiate(
            cfg.benchmark,
            controller=controller,
            sage_runtime=runtime,
            logger=logger,
        )
        try:
            metrics = runner.run()
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            logger.finish_run(status="completed")
            return
        except Exception:
            logger.finish_run(status="failed")
            raise

    if mode == "test":
        print("Running in test mode: the agent execution is tested.")
        print(model.invoke("What is a transformer model in NLP?"))
        return

    raise ValueError(f"Unsupported mode: {mode!r}. Use 'chat' or 'benchmark'.")


if __name__ == "__main__":
    main()
