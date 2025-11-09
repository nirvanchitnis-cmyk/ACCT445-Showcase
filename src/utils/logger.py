"""Centralized logging utilities for the ACCT445 Showcase project."""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class JSONFormatter(logging.Formatter):
    """Format log records as JSON for ingestion by log pipelines."""

    _reserved = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        extras = {key: value for key, value in record.__dict__.items() if key not in self._reserved}
        if extras:
            payload["extras"] = extras

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


def _build_formatter(json_format: bool) -> logging.Formatter:
    if json_format:
        return JSONFormatter()
    return logging.Formatter(fmt=DEFAULT_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(
    name: str,
    *,
    level: int = logging.INFO,
    log_file: Path | None = None,
    json_format: bool = False,
    rotate: bool = True,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
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
    json_format:
        Emit JSON lines for easier parsing in production.
    rotate:
        Use :class:`RotatingFileHandler` for file outputs.
    max_bytes:
        Maximum bytes per log file before roll-over occurs.
    backup_count:
        Number of rotated log files to keep.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = _build_formatter(json_format)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if rotate:
            file_handler: logging.Handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter if json_format else _build_formatter(False))
        logger.addHandler(file_handler)

    return logger
