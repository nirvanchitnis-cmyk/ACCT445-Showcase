"""
Data Loader Utilities

Lightweight data loading for CNOI scores and market returns.
No large file dependencies - all data fetched on-demand.
"""

from datetime import timedelta

import pandas as pd
import yfinance as yf

from src.utils.exceptions import DataDownloadError, DataValidationError
from src.utils.logger import get_logger
from src.utils.validation import validate_cnoi_schema, validate_returns_schema

logger = get_logger(__name__)


def load_cnoi_data(filepath: str) -> pd.DataFrame:
    """
    Load CNOI scores from CSV.

    Args:
        filepath: Path to CNOI CSV file

    Returns:
        DataFrame with columns: cik, ticker, filing_date, CNOI, D, G, R, J, T, S, X, issuer

    Example:
        >>> cnoi = load_cnoi_data('config/sample_cnoi.csv')
        >>> cnoi[['ticker', 'CNOI', 'filing_date']].head()
    """
    df = pd.read_csv(filepath)

    if "filing_date" in df.columns:
        df["filing_date"] = pd.to_datetime(df["filing_date"])

    if "filing_date" in df.columns and "quarter" not in df.columns:
        df["quarter"] = df["filing_date"].dt.to_period("Q")

    validate_cnoi_schema(df)

    logger.info("Loaded %s CNOI rows from %s", len(df), filepath)
    logger.debug(
        "CNOI summary — CIKs: %s, range: %s → %s, CNOI range %.2f → %.2f",
        df["cik"].nunique(),
        df["filing_date"].min(),
        df["filing_date"].max(),
        df["CNOI"].min(),
        df["CNOI"].max(),
    )

    return df


def load_market_returns(
    tickers: list[str], start: str, end: str, frequency: str = "daily"
) -> pd.DataFrame:
    """
    Download stock returns from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        frequency: 'daily', 'weekly', 'monthly'

    Returns:
        DataFrame with columns: date, ticker, price, return, volume

    Example:
        >>> tickers = ['WFC', 'JPM', 'BAC']
        >>> returns = load_market_returns(tickers, '2023-01-01', '2025-11-01')
        >>> returns.head()
    """
    if not tickers:
        raise DataValidationError("Ticker list must contain at least one symbol.")

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    if start_dt >= end_dt:
        raise DataValidationError("`start` must be earlier than `end`.")

    logger.info(
        "Downloading %s tickers from %s to %s",
        len(tickers),
        start_dt.date(),
        end_dt.date(),
    )

    raw = yf.download(
        tickers=tickers,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        group_by="ticker",
        progress=False,
        threads=False,
    )

    if raw.empty:
        raise DataDownloadError("yfinance returned an empty dataframe.")

    def _prepare_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
        if frame.empty:
            logger.warning("No data returned for %s", ticker)
            return None

        price_col = next(
            (col for col in ("Adj Close", "Close") if col in frame.columns),
            None,
        )
        if price_col is None:
            logger.warning("Missing price column for %s", ticker)
            return None

        prices = frame[price_col].rename("price")
        returns = prices.pct_change().rename("return")
        volume = frame.get("Volume", pd.Series(index=frame.index, dtype=float)).rename("volume")

        tidy = pd.DataFrame(
            {
                "date": frame.index,
                "ticker": ticker,
                "price": prices.values,
                "return": returns.values,
                "volume": volume.values,
            }
        ).dropna(subset=["return"])

        return tidy

    prepared_frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        available = set(raw.columns.get_level_values(1))
        for ticker in tickers:
            if ticker not in available:
                logger.warning("Ticker %s missing from download output", ticker)
                continue
            ticker_frame = raw.xs(ticker, axis=1, level=1)
            prepared = _prepare_frame(ticker_frame, ticker)
            if prepared is not None:
                prepared_frames.append(prepared)
    else:
        ticker = tickers[0]
        prepared = _prepare_frame(raw, ticker)
        if prepared is not None:
            prepared_frames.append(prepared)

    if not prepared_frames:
        raise DataDownloadError("No ticker data was processed successfully.")

    df = pd.concat(prepared_frames, ignore_index=True)

    if frequency in ("weekly", "monthly"):
        rule = "W" if frequency == "weekly" else "M"
        df = (
            df.set_index("date")
            .groupby("ticker")
            .resample(rule)
            .agg(
                {
                    "price": "last",
                    "return": lambda x: (1 + x).prod() - 1,
                    "volume": "sum",
                }
            )
            .reset_index()
        )

    validate_returns_schema(df, required_columns=("date", "ticker", "return"))
    logger.info(
        "Downloaded %s observations spanning %s tickers",
        len(df),
        df["ticker"].nunique(),
    )

    return df


def compute_forward_returns(
    returns_df: pd.DataFrame, horizon: int = 1, frequency: str = "quarterly"
) -> pd.DataFrame:
    """
    Compute forward returns for backtesting.

    Args:
        returns_df: DataFrame from load_market_returns()
        horizon: Number of periods forward (1 = next quarter)
        frequency: 'daily', 'weekly', 'monthly', 'quarterly'

    Returns:
        DataFrame with added 'ret_fwd' column

    Example:
        >>> returns = load_market_returns(['WFC'], '2023-01-01', '2025-11-01')
        >>> fwd = compute_forward_returns(returns, horizon=1, frequency='quarterly')
    """
    if horizon <= 0:
        raise DataValidationError("`horizon` must be a positive integer.")

    validate_returns_schema(returns_df, required_columns=("date", "ticker", "return"))
    df = returns_df.copy()

    # Resample to target frequency if needed
    if frequency == "quarterly":
        df["period"] = pd.PeriodIndex(df["date"], freq="Q")
        period_returns = (
            df.groupby(["ticker", "period"])["return"]
            .apply(lambda x: (1 + x).prod() - 1)
            .reset_index()
        )
        period_returns = period_returns.rename(columns={"return": "ret_period"})

        # Compute forward returns
        period_returns["ret_fwd"] = period_returns.groupby("ticker")["ret_period"].shift(-horizon)

        if "ret_fwd" in df.columns:
            df = df.drop(columns=["ret_fwd"])

        df = df.merge(
            period_returns[["ticker", "period", "ret_fwd"]], on=["ticker", "period"], how="left"
        )

    else:
        # For daily/weekly/monthly, simple shift
        df["ret_fwd"] = df.groupby("ticker")["return"].shift(-horizon)

    return df


def merge_cnoi_with_returns(
    cnoi_df: pd.DataFrame, returns_df: pd.DataFrame, lag_days: int = 1
) -> pd.DataFrame:
    """
    Merge CNOI scores with forward returns.

    Enforces information timing: CNOI score from filing_date t
    predicts returns starting t + lag_days.

    Args:
        cnoi_df: DataFrame from load_cnoi_data()
        returns_df: DataFrame from load_market_returns()
        lag_days: Days after filing before signal is tradable

    Returns:
        Merged DataFrame with CNOI + forward returns

    Example:
        >>> cnoi = load_cnoi_data('config/sample_cnoi.csv')
        >>> returns = load_market_returns(cnoi['ticker'].unique(), '2023-01-01', '2025-11-01')
        >>> merged = merge_cnoi_with_returns(cnoi, returns, lag_days=1)
    """
    if "ticker" not in cnoi_df.columns or "filing_date" not in cnoi_df.columns:
        raise DataValidationError("CNOI dataframe must include `ticker` and `filing_date`.")

    validate_returns_schema(returns_df, required_columns=("date", "ticker", "return"))

    # Ensure tickers match
    common_tickers = set(cnoi_df["ticker"].dropna()) & set(returns_df["ticker"])
    logger.info("Found %s overlapping tickers between CNOI and returns", len(common_tickers))

    cnoi_sub = cnoi_df[cnoi_df["ticker"].isin(common_tickers)].copy()
    returns_sub = returns_df[returns_df["ticker"].isin(common_tickers)].copy()

    # Apply lag to filing_date
    cnoi_sub["decision_date"] = cnoi_sub["filing_date"] + timedelta(days=lag_days)

    # Merge on ticker + nearest date
    merged = pd.merge_asof(
        cnoi_sub.sort_values("decision_date"),
        returns_sub.sort_values("date"),
        left_on="decision_date",
        right_on="date",
        by="ticker",
        direction="forward",
        tolerance=pd.Timedelta("30 days"),
    )

    logger.info("Merged %s observations after lag alignment", len(merged))
    logger.debug("Missing returns count: %s", merged["return"].isna().sum())

    return merged


def create_sample_cnoi_file(
    full_cnoi_path: str, output_path: str, n_top: int = 20, n_bottom: int = 20
) -> None:
    """
    Create lightweight sample CNOI file (top + bottom banks only).

    Args:
        full_cnoi_path: Path to full CNOI CSV
        output_path: Where to save sample
        n_top: Number of most transparent banks
        n_bottom: Number of most opaque banks

    Example:
        >>> create_sample_cnoi_file(
        ...     '/path/to/cnoi_full.csv',
        ...     'config/sample_cnoi.csv',
        ...     n_top=20,
        ...     n_bottom=20
        ... )
    """
    df = pd.read_csv(full_cnoi_path)

    # Get latest CNOI per bank
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    latest = df.sort_values("filing_date").groupby("cik").last().reset_index()

    # Top N (most transparent)
    top = latest.nsmallest(n_top, "CNOI")

    # Bottom N (most opaque)
    bottom = latest.nlargest(n_bottom, "CNOI")

    # Combine
    sample = pd.concat([top, bottom])

    # Get all historical filings for these banks
    sample_ciks = sample["cik"].unique()
    full_sample = df[df["cik"].isin(sample_ciks)]

    full_sample.to_csv(output_path, index=False)
    logger.info("Created sample CNOI file: %s", output_path)
    logger.info(
        "Sample coverage — Banks: %s, Filings: %s (%.1f%% of original)",
        len(sample_ciks),
        len(full_sample),
        len(full_sample) * 100 / len(df),
    )


if __name__ == "__main__":  # pragma: no cover
    # Demo
    logger.info("=" * 60)
    logger.info("Data Loader Demo")
    logger.info("=" * 60)

    # Create sample file if full CNOI exists
    import os

    full_path = "../../../ACCT445-Banks/out/cnoi_top100_20251101120954.csv"
    if os.path.exists(full_path):
        create_sample_cnoi_file(full_path, "config/sample_cnoi.csv", n_top=20, n_bottom=20)
    else:
        logger.warning("Full CNOI file not found: %s", full_path)
        logger.warning("Skipping sample creation")
