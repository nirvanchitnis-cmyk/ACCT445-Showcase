"""Alerting utilities for the automated runner."""

from __future__ import annotations

from src.utils.config import get_config_value
from src.utils.logger import get_logger

LOGGER = get_logger(__name__)


def send_alert(subject: str, message: str) -> None:
    """
    Emit an alert message and optionally trigger downstream integrations.

    Args:
        subject: Short alert title (e.g., ``\"Data quality issue\"``)
        message: Human-readable context

    Example:
        >>> send_alert(\"Data coverage\", \"Market fetch returned < 80% coverage\")
    """

    LOGGER.warning("ALERT: %s - %s", subject, message)

    if bool(get_config_value("alerts.enable_email", False)):
        LOGGER.info("Email alerts not yet configured; subject=%s", subject)
