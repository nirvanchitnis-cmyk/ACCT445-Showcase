"""Placeholder daily backtest runner.

This stub keeps the container alive until the full automation suite arrives
in Task 5.2. It simply logs a reminder and sleeps.
"""

from __future__ import annotations

import time
from pathlib import Path

from src.utils.logger import get_logger

LOGGER = get_logger(__name__, log_file=Path("logs/runner_placeholder.log"))


def main() -> None:
    """Log a placeholder message and sleep indefinitely."""
    LOGGER.warning("Daily backtest runner placeholder loaded. Full automation pending Task 5.2.")
    LOGGER.info("Sleeping to keep container healthy...")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
