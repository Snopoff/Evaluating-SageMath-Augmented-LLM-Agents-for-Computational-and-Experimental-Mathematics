import logging
from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def progress(enabled: bool, message: str) -> None:
    if enabled:
        console.print(f"[cyan]progress[/cyan] {message}")
