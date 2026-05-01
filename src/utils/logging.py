"""Logging utilities for the object detection project."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name."""
    return logging.getLogger(name)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging format and level for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
