"""
Production-grade scheduler with:
- DST-aware cron scheduling (APScheduler)
- Job locks (prevent duplicate runs)
- Graceful shutdown
- Error recovery
"""

import signal
import sys
from pathlib import Path

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from filelock import FileLock, Timeout

from src.runner.daily_backtest import run_daily_update
from src.utils.config import get_config_value
from src.utils.logger import get_logger

logger = get_logger(__name__, log_file=Path("logs/scheduler.log"), json_format=True)


class ProductionScheduler:
    """Scheduler with job locks and graceful shutdown."""

    def __init__(self, lock_file: Path = Path("/tmp/acct445_runner.lock")):
        self.lock_file = lock_file
        self.scheduler = None
        self.running = True

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.scheduler:
            self.scheduler.shutdown(wait=True)
        sys.exit(0)

    def run_with_lock(self):
        """
        Run daily backtest with file lock (prevent concurrent runs).

        If another process is running, skip this execution.
        """
        try:
            with FileLock(self.lock_file, timeout=1):
                logger.info("Acquired job lock, starting daily backtest...")
                run_daily_update()
                logger.info("Daily backtest completed successfully")
        except Timeout:
            logger.warning("Another backtest is already running (lock held), skipping...")
        except Exception as e:
            logger.error(f"Daily backtest failed: {e}", exc_info=True)

    def start(self):
        """Start scheduler with DST-aware cron."""
        # Get config
        schedule_time = get_config_value("runner.schedule_time", "18:00")  # 6 PM ET
        timezone = pytz.timezone("America/New_York")

        # Parse schedule time
        hour, minute = map(int, schedule_time.split(":"))

        # Create scheduler
        self.scheduler = BlockingScheduler(timezone=timezone)

        # Add job with cron trigger (handles DST automatically)
        trigger = CronTrigger(hour=hour, minute=minute, timezone=timezone)
        self.scheduler.add_job(
            self.run_with_lock,
            trigger=trigger,
            id="daily_backtest",
            name="Daily Backtest Update",
            misfire_grace_time=3600,  # Allow 1 hour late execution
            coalesce=True,  # If multiple misfires, run only once
            max_instances=1,  # Only one instance at a time
        )

        logger.info(
            f"Scheduler started. Daily backtest scheduled at {schedule_time} ET (timezone-aware)"
        )
        logger.info(f"Next run: {self.scheduler.get_jobs()[0].next_run_time}")

        # Run immediately on startup (optional)
        run_on_startup = get_config_value("runner.run_on_startup", True)
        if run_on_startup:
            logger.info("Running initial backtest on startup...")
            self.run_with_lock()

        # Start blocking scheduler
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped by user")


def main():
    """Entry point for scheduler."""
    scheduler = ProductionScheduler()
    scheduler.start()


if __name__ == "__main__":
    main()
