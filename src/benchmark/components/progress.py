from __future__ import annotations

from typing import Callable

from src.benchmark.components.config import BenchmarkConfig
from src.utils.console_logging import ConsoleLogger


def build_progress_logger(config: BenchmarkConfig, logger: ConsoleLogger | None = None) -> Callable[[str], None]:
    def progress(message: str) -> None:
        if not config.progress_logs:
            return
        if logger is not None:
            logger.progress(f"[benchmark] {message}")
            return
        print(f"[benchmark] {message}", flush=True)

    return progress
