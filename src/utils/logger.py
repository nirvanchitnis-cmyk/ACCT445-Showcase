"""Centralized logging utilities for the ACCT445 Showcase project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Return a configured logger instance.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__``.
    level:
        Minimum severity level for the console handler.
    log_file:
        Optional file to duplicate log output for post-run diagnostics.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
