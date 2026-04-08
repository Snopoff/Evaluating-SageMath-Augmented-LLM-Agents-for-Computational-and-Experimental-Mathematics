from pathlib import Path

import json
import hydra.utils as hu
from omegaconf import DictConfig, OmegaConf

from src.utils.console_logging import ConsoleLogger


def resolve_text_asset(asset: DictConfig, *, label: str, logger: ConsoleLogger | None = None) -> str:
    text = OmegaConf.select(asset, "text")
    file_value = OmegaConf.select(asset, "file")

    has_text = isinstance(text, str) and text.strip()
    has_file = isinstance(file_value, str) and file_value.strip()

    if bool(has_text) == bool(has_file):
        raise ValueError(
            f"Config '{label}' must define exactly one of 'text' or 'file'. But got: "
            + json.dumps(OmegaConf.to_container(asset), indent=2)
        )

    if has_text:
        return text.strip()

    prompt_path = Path(hu.to_absolute_path(file_value))
    if logger is not None:
        logger.progress(f"loading {label} from file: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8").strip()


def resolve_prompt(prompt: DictConfig, logger: ConsoleLogger | None = None) -> str:
    return resolve_text_asset(prompt, label="prompt", logger=logger)
