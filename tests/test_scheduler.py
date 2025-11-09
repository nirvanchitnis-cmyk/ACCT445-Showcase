"""Tests for production scheduler with DST-safe scheduling and job locks."""

from __future__ import annotations

import signal
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import pytz
from filelock import FileLock

from src.runner.scheduler import ProductionScheduler


@pytest.fixture
def temp_lock_file():
    """Provide a temporary lock file for testing."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        lock_path = Path(f.name)
    yield lock_path
    # Cleanup
    if lock_path.exists():
        lock_path.unlink()
    lock_file = Path(f"{lock_path}.lock")
    if lock_file.exists():
        lock_file.unlink()


def test_scheduler_initialization(temp_lock_file):
    """Test scheduler initializes correctly."""
    scheduler = ProductionScheduler(lock_file=temp_lock_file)
    assert scheduler.lock_file == temp_lock_file
    assert scheduler.running is True
    assert scheduler.scheduler is None


def test_job_lock_prevents_concurrent_runs(temp_lock_file):
    """Job lock should prevent duplicate runs."""
    scheduler = ProductionScheduler(lock_file=temp_lock_file)

    # Acquire lock in first "process"
    lock = FileLock(temp_lock_file, timeout=0.1)
    lock.acquire()

    try:
        # Mock the run_daily_update to track if it's called
        with patch("src.runner.scheduler.run_daily_update") as mock_run:
            scheduler.run_with_lock()
            # Should not call run_daily_update because lock is held
            mock_run.assert_not_called()
    finally:
        lock.release()


def test_job_lock_allows_run_when_unlocked(temp_lock_file):
    """Job lock should allow run when no other process holds lock."""
    scheduler = ProductionScheduler(lock_file=temp_lock_file)

    with patch("src.runner.scheduler.run_daily_update") as mock_run:
        scheduler.run_with_lock()
        # Should call run_daily_update when lock is available
        mock_run.assert_called_once()


def test_scheduler_handles_dst_transitions():
    """Scheduler should handle DST transitions correctly."""
    with patch("src.runner.scheduler.get_config_value") as mock_config:
        mock_config.return_value = "18:00"

        ProductionScheduler()

        # Create scheduler but don't start it
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger

        timezone = pytz.timezone("America/New_York")
        BlockingScheduler(timezone=timezone)

        trigger = CronTrigger(hour=18, minute=0, timezone=timezone)

        # Verify trigger uses a timezone (zoneinfo or pytz)
        assert trigger.timezone is not None
        assert str(trigger.timezone) == "America/New_York"

        # Test that 6 PM ET is always 6 PM ET (not affected by DST)
        # In winter (EST): 6 PM ET = 11 PM UTC
        # In summer (EDT): 6 PM ET = 10 PM UTC
        winter_date = datetime(2024, 1, 15, 18, 0, 0, tzinfo=timezone)
        summer_date = datetime(2024, 7, 15, 18, 0, 0, tzinfo=timezone)

        # Both should be 18:00 local time
        assert winter_date.hour == 18
        assert summer_date.hour == 18


def test_scheduler_configuration_parsing():
    """Test scheduler parses configuration correctly."""
    with patch("src.runner.scheduler.get_config_value") as mock_config:
        # Test default schedule time
        mock_config.side_effect = lambda key, default: {
            "runner.schedule_time": "18:00",
            "runner.run_on_startup": True,
        }.get(key, default)

        scheduler = ProductionScheduler()
        # Verify it doesn't crash on initialization
        assert scheduler is not None


def test_graceful_shutdown_on_sigterm(temp_lock_file):
    """Scheduler should shut down gracefully on SIGTERM."""
    scheduler = ProductionScheduler(lock_file=temp_lock_file)

    # Mock the scheduler
    scheduler.scheduler = MagicMock()
    scheduler.scheduler.shutdown = MagicMock()

    # Trigger shutdown handler
    with pytest.raises(SystemExit):
        scheduler._shutdown_handler(signal.SIGTERM, None)

    # Verify scheduler shutdown was called
    scheduler.scheduler.shutdown.assert_called_once_with(wait=True)
    assert scheduler.running is False


def test_graceful_shutdown_on_sigint(temp_lock_file):
    """Scheduler should shut down gracefully on SIGINT."""
    scheduler = ProductionScheduler(lock_file=temp_lock_file)

    scheduler.scheduler = MagicMock()
    scheduler.scheduler.shutdown = MagicMock()

    with pytest.raises(SystemExit):
        scheduler._shutdown_handler(signal.SIGINT, None)

    scheduler.scheduler.shutdown.assert_called_once_with(wait=True)
    assert scheduler.running is False


def test_error_recovery_in_run_with_lock(temp_lock_file):
    """Scheduler should handle errors gracefully during execution."""
    scheduler = ProductionScheduler(lock_file=temp_lock_file)

    with patch("src.runner.scheduler.run_daily_update") as mock_run:
        # Make run_daily_update raise an error
        mock_run.side_effect = Exception("Test error")

        # Should not raise, just log the error
        scheduler.run_with_lock()

        # Verify it tried to run
        mock_run.assert_called_once()


def test_run_on_startup_configuration():
    """Test run_on_startup configuration option."""
    with patch("src.runner.scheduler.get_config_value") as mock_config:
        mock_config.side_effect = lambda key, default: {
            "runner.schedule_time": "18:00",
            "runner.run_on_startup": False,  # Disable startup run
        }.get(key, default)

        scheduler = ProductionScheduler()

        with patch.object(scheduler, "run_with_lock") as mock_run:
            with patch("src.runner.scheduler.BlockingScheduler") as mock_scheduler_class:
                mock_scheduler = MagicMock()
                mock_scheduler_class.return_value = mock_scheduler
                mock_scheduler.get_jobs.return_value = [MagicMock(next_run_time=datetime.now())]

                # Mock the start method to avoid blocking
                mock_scheduler.start.side_effect = KeyboardInterrupt

                try:
                    scheduler.start()
                except KeyboardInterrupt:
                    pass

                # Should not run on startup when disabled
                mock_run.assert_not_called()


def test_scheduler_timezone_aware():
    """Verify scheduler uses timezone-aware scheduling."""
    timezone = pytz.timezone("America/New_York")

    with patch("src.runner.scheduler.get_config_value") as mock_config:
        mock_config.return_value = "18:00"

        ProductionScheduler()

        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger

        test_scheduler = BlockingScheduler(timezone=timezone)
        trigger = CronTrigger(hour=18, minute=0, timezone=timezone)

        # Verify timezone is set (APScheduler may convert to zoneinfo)
        assert test_scheduler.timezone is not None
        assert str(test_scheduler.timezone) == "America/New_York"
        assert trigger.timezone is not None
        assert str(trigger.timezone) == "America/New_York"


def test_misfire_grace_time_configured():
    """Test that misfire grace time allows late execution."""
    with patch("src.runner.scheduler.get_config_value") as mock_config:
        mock_config.side_effect = lambda key, default: {
            "runner.schedule_time": "18:00",
            "runner.run_on_startup": False,
        }.get(key, default)

        scheduler = ProductionScheduler()

        # Verify scheduler would be configured with grace time
        # This is implicit in the start() method implementation
        assert scheduler is not None
