import logging
from rich.console import Console
from rich.logging import RichHandler

# `record=True` allows rich to keep all printed messages.
console = Console(record=True)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )


def save_logs(log_name: str, html=True) -> None:
    # Save output with colors preserved
    if html:
        console.save_html(log_name)
    else:
        console.save_text(log_name)


def progress(message: str) -> None:
    console.print(f"[cyan]\[progress][/cyan] {message}")  # type: ignore
