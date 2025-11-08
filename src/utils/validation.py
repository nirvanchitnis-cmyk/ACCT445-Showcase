"""Lightweight data validation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _require_columns(df: pd.DataFrame, columns: Sequence[str], df_name: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise DataValidationError(f"{df_name} missing required columns: {sorted(missing)}")


def validate_cnoi_schema(df: pd.DataFrame) -> None:
    """Validate that a CNOI dataframe looks as expected."""

    required_cols = [
        "cik",
        "filing_date",
        "CNOI",
        "D",
        "G",
        "R",
        "J",
        "T",
        "S",
        "X",
    ]
    _require_columns(df, required_cols, "CNOI data")

    if not pd.api.types.is_datetime64_any_dtype(df["filing_date"]):
        raise DataValidationError("CNOI data must have datetime `filing_date`.")

    if df.empty:
        raise DataValidationError("CNOI data is empty.")

    duplicates = df.duplicated(subset=["cik", "filing_date"]).sum()
    if duplicates:
        logger.warning("Detected %s duplicate (cik, filing_date) rows.", duplicates)

    logger.info("CNOI schema validation passed (%s rows).", len(df))


def validate_returns_schema(
    df: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
) -> None:
    """Validate a generic returns dataframe."""

    required = list(required_columns or ("date", "ticker", "return"))
    _require_columns(df, required, "Returns data")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise DataValidationError("Returns data must use datetime `date` column.")

    if df.empty:
        raise DataValidationError("Returns data is empty.")

    logger.debug("Returns schema validation passed (%s rows).", len(df))


def validate_event_inputs(
    returns_df: pd.DataFrame,
    market_returns: pd.Series,
    *,
    required_stock_cols: Sequence[str] | None = None,
) -> None:
    """Validate inputs for the event study pipeline."""

    required_stock_cols = required_stock_cols or ("date", "ticker", "return")
    _require_columns(returns_df, required_stock_cols, "Event study returns")

    if returns_df.empty:
        raise DataValidationError("Event study returns frame is empty.")

    if market_returns.empty:
        raise DataValidationError("Market returns series is empty.")

    if not pd.api.types.is_datetime64_any_dtype(returns_df["date"]):
        raise DataValidationError("Event study returns `date` must be datetime.")

    if not isinstance(market_returns.index, pd.DatetimeIndex):
        raise DataValidationError("Market returns must be indexed by datetime.")

    logger.debug(
        "Event inputs validated: %s stock rows, %s market days.",
        len(returns_df),
        len(market_returns),
    )
