"""Tests for validation and logging utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger
from src.utils.validation import (
    validate_cnoi_schema,
    validate_event_inputs,
    validate_returns_schema,
)


class TestLogger:
    """Tests for logger helper."""

    def test_logger_creates_file(self, tmp_path: Path):
        log_path = tmp_path / "logs" / "test.log"
        logger = get_logger("acct445.tests.logger", log_file=log_path)
        logger.info("hello world")

        assert log_path.exists()
        logger.info("second message")  # ensure handler persists

        # Clean up handlers to avoid leakage across tests
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    def test_logger_reuses_existing_instance(self, tmp_path: Path):
        name = "acct445.tests.reuse"
        logger = get_logger(name, log_file=tmp_path / "reuse.log")
        handler_count = len(logger.handlers)
        logger_again = get_logger(name)
        assert logger is logger_again
        assert len(logger_again.handlers) == handler_count
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


class TestValidation:
    """Tests for dataframe schema validation."""

    def test_validate_cnoi_schema_success(self, sample_cnoi_data: pd.DataFrame):
        validate_cnoi_schema(sample_cnoi_data)

    def test_validate_cnoi_schema_missing(self, sample_cnoi_data: pd.DataFrame):
        bad = sample_cnoi_data.drop(columns=["CNOI"])
        with pytest.raises(DataValidationError):
            validate_cnoi_schema(bad)

    def test_validate_returns_schema(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="D"),
                "ticker": ["A", "A", "A"],
                "return": [0.01, -0.02, 0.03],
            }
        )
        validate_returns_schema(df)

    def test_validate_returns_schema_missing_date_type(self):
        df = pd.DataFrame(
            {
                "date": ["2023-01-01"],
                "ticker": ["AAPL"],
                "return": [0.01],
            }
        )
        with pytest.raises(DataValidationError):
            validate_returns_schema(df)

    def test_validate_event_inputs_errors(self, sample_event_study_returns: pd.DataFrame):
        market = pd.Series([], dtype=float)
        with pytest.raises(DataValidationError):
            validate_event_inputs(sample_event_study_returns, market)
