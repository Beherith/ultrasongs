"""Logging configuration for the CLI."""

import logging
import sys
from datetime import datetime


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure root logger with stdout handler.

    Args:
        verbose: Set level to DEBUG.
        quiet: Set level to WARNING.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Override formatTime to use our date format
    formatter.default_time_format = "%Y-%m-%d %H:%M:%S"
    formatter.default_msec_format = "%s.%03d"

    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a module-specific logger."""
    return logging.getLogger(name)
