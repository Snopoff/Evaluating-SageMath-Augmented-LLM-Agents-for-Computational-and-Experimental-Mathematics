from pathlib import Path

import json
import hydra.utils as hu
from omegaconf import DictConfig, OmegaConf

from src.utils.logging import progress


def resolve_prompt(prompt: DictConfig, progress_logs: bool) -> str:
    text = OmegaConf.select(prompt, "text")
    file_value = OmegaConf.select(prompt, "file")

    has_text = isinstance(text, str) and text.strip()
    has_file = isinstance(file_value, str) and file_value.strip()

    if bool(has_text) == bool(has_file):
        raise ValueError(
            "Config 'prompt' must define exactly one of 'text' or 'file'. But got: " + json.dumps(OmegaConf.to_container(prompt), indent=2)
        )

    if has_text:
        return text.strip()

    prompt_path = Path(hu.to_absolute_path(file_value))
    if progress_logs:
        progress(f"loading prompt from file: {prompt_path}")

    return prompt_path.read_text(encoding="utf-8").strip()
