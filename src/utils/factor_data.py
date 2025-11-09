"""Fama-French factor data downloader with caching and DVC versioning."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from pandas_datareader import data as web

from src.utils.caching import disk_cache
from src.utils.logger import get_logger

logger = get_logger(__name__)


@disk_cache(cache_dir="data/cache/factors", disable_cache_kwarg="disable_cache")
def fetch_fama_french_factors(
    factor_set: str = "F-F_Research_Data_5_Factors_2x3_daily",
    start_date: str = "2023-01-01",
    end_date: str = "2025-12-31",
    disable_cache: bool = False,
) -> pd.DataFrame:
    """
    Fetch Fama-French factors from Ken French data library.

    Args:
        factor_set: One of:
            - "F-F_Research_Data_Factors_daily" (FF3: Mkt-RF, SMB, HML)
            - "F-F_Research_Data_5_Factors_2x3_daily" (FF5: + RMW, CMA)
            - "F-F_Momentum_Factor_daily" (MOM)
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format
        disable_cache: Force fresh download if True

    Returns:
        DataFrame with columns: ['Mkt-RF', 'SMB', 'HML', 'RF'] (FF3)
                            or ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'] (FF5)
                            or ['MOM'] (Momentum)
        Index: DatetimeIndex (business days)
        Values: Daily returns as decimals (not percentages)

    Example:
        >>> factors = fetch_fama_french_factors("F-F_Research_Data_5_Factors_2x3_daily")
        >>> factors.head()
    """
    logger.info("Fetching %s from Ken French library (%s to %s)", factor_set, start_date, end_date)

    try:
        # Fetch from pandas-datareader (returns dictionary with keys 0, 1, ...)
        # Key 0 is the main data table
        raw_data = web.DataReader(factor_set, "famafrench", start=start_date, end=end_date)

        # Extract the first DataFrame (main factor data)
        if isinstance(raw_data, dict):
            df = raw_data[0]
        else:
            df = raw_data

        # Convert from percentage to decimal (French data is in %)
        df = df / 100.0

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Align to business days only
        df = df.sort_index()

        logger.info(
            "Successfully fetched %s rows, %s columns: %s",
            len(df),
            len(df.columns),
            df.columns.tolist(),
        )

        return df

    except Exception as e:
        logger.error("Failed to fetch %s: %s", factor_set, e)
        raise


def fetch_all_factors(
    start_date: str = "2023-01-01", end_date: str = "2025-12-31", disable_cache: bool = False
) -> pd.DataFrame:
    """
    Fetch FF5 + Momentum + RF in one call. Merge on date.

    Args:
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        disable_cache: Force fresh download if True

    Returns:
        DataFrame with columns: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF', 'MOM']
        Index: DatetimeIndex (business days)

    Example:
        >>> all_factors = fetch_all_factors("2023-01-01", "2024-12-31")
        >>> all_factors[['Mkt-RF', 'MOM']].head()
    """
    logger.info("Fetching all factors (FF5 + Momentum) from %s to %s", start_date, end_date)

    # Fetch FF5 (includes RF)
    ff5 = fetch_fama_french_factors(
        "F-F_Research_Data_5_Factors_2x3_daily",
        start_date=start_date,
        end_date=end_date,
        disable_cache=disable_cache,
    )

    # Fetch Momentum
    mom = fetch_fama_french_factors(
        "F-F_Momentum_Factor_daily",
        start_date=start_date,
        end_date=end_date,
        disable_cache=disable_cache,
    )

    # Merge on date (inner join to handle any date mismatches)
    merged = ff5.join(mom, how="inner")

    logger.info(
        "Merged factors: %s rows, %s columns: %s",
        len(merged),
        len(merged.columns),
        merged.columns.tolist(),
    )

    return merged


def save_factors_to_dvc(df: pd.DataFrame, output_path: Path | str) -> None:
    """
    Save factors CSV and add to DVC tracking.

    Args:
        df: Factor DataFrame to save
        output_path: Path to save CSV (e.g., 'data/factors/ff5_momentum_daily.csv')

    Example:
        >>> factors = fetch_all_factors()
        >>> save_factors_to_dvc(factors, 'data/factors/ff5_momentum_daily.csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV
    df.to_csv(output_path)
    logger.info("Saved factors to %s (%s rows)", output_path, len(df))

    # Add to DVC if available
    try:
        result = subprocess.run(
            ["dvc", "add", str(output_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0:
            logger.info("Added %s to DVC tracking", output_path)
            logger.debug("DVC output: %s", result.stdout)
        else:
            logger.warning("DVC add failed (code %s): %s", result.returncode, result.stderr)
            logger.info("File saved but not tracked by DVC")

    except FileNotFoundError:
        logger.warning("DVC not installed; skipping DVC tracking")
    except Exception as e:
        logger.warning("DVC tracking failed: %s", e)


def load_factors_from_csv(csv_path: Path | str) -> pd.DataFrame:
    """
    Load factors from CSV with proper date parsing.

    Args:
        csv_path: Path to factors CSV

    Returns:
        DataFrame with DatetimeIndex

    Example:
        >>> factors = load_factors_from_csv('data/factors/ff5_momentum_daily.csv')
        >>> factors.head()
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Factor file not found: {csv_path}")

    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    logger.info("Loaded factors from %s (%s rows)", csv_path, len(df))

    return df


__all__ = [
    "fetch_fama_french_factors",
    "fetch_all_factors",
    "save_factors_to_dvc",
    "load_factors_from_csv",
]
