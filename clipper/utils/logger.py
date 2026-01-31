"""Logging utilities."""

import logging
import sys
from rich.logging import RichHandler
from rich.console import Console

console = Console()

def setup_logging(level: int = logging.INFO) -> None:
    """Setup rich logging handler."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
