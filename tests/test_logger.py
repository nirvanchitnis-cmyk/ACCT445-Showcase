"""Tests for structured logging utilities."""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler

from src.utils.logger import JSONFormatter, get_logger


def test_json_formatter_emits_valid_json(tmp_path):
    """JSON formatter should emit parseable structured logs with extras."""
    log_file = tmp_path / "structured.log"
    logger = get_logger(
        "acct445.json_test",
        log_file=log_file,
        json_format=True,
        level=10,
    )
    logger.info("runner update complete", extra={"ticker": "BANK1"})

    for handler in logger.handlers:
        handler.flush()

    lines = log_file.read_text().strip().splitlines()
    assert lines
    payload = json.loads(lines[-1])
    assert payload["message"] == "runner update complete"
    assert payload["extras"]["ticker"] == "BANK1"
    assert payload["level"] == "INFO"


def test_rotating_handler_configured(tmp_path):
    """File logging should default to a rotating handler for production safety."""
    log_file = tmp_path / "rotating.log"
    logger = get_logger("acct445.rotating_test", log_file=log_file, rotate=True)
    rotating_handlers = [
        handler for handler in logger.handlers if isinstance(handler, RotatingFileHandler)
    ]
    assert rotating_handlers, "expected a RotatingFileHandler to be attached"


def test_json_formatter_handles_exceptions():
    """Ensure JSONFormatter can render exception info."""
    formatter = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.getLogger("acct445.exc").makeRecord(
            "acct445.exc",
            logging.ERROR,
            __file__,
            10,
            "failure",
            args=(),
            exc_info=sys.exc_info(),
            func=None,
            extra=None,
        )
    payload = json.loads(formatter.format(record))
    assert payload["level"] == "ERROR"
    assert "exc_info" in payload
