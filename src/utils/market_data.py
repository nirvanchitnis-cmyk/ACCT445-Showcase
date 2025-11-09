"""Robust yfinance integration with caching, retries, and validation."""

from __future__ import annotations

import time
from collections.abc import Iterable
from functools import wraps
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils.exceptions import DataDownloadError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "yfinance"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 24.0
EXTREME_RETURN_THRESHOLD = 0.30


def rate_limited(calls_per_second: float = 2.0):
    """Decorator that enforces a minimum interval between function calls."""

    if calls_per_second <= 0:
        raise ValueError("calls_per_second must be positive")

    min_interval = 1.0 / calls_per_second

    def decorator(func):
        last_called = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                logger.debug(
                    "Rate limiting %s; sleeping %.3fs to respect API constraints.",
                    func.__name__,
                    wait_time,
                )
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result

        return wrapper

    return decorator


def _cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    safe_ticker = ticker.upper().replace("/", "-")
    return CACHE_DIR / f"{safe_ticker}_{start_date}_{end_date}.pkl"


def _cache_is_fresh(cache_file: Path) -> bool:
    if not cache_file.exists():
        return False
    age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
    return age_hours < CACHE_TTL_HOURS


def _load_cache(cache_file: Path) -> pd.DataFrame | None:
    try:
        return pd.read_pickle(cache_file)
    except (OSError, ValueError):  # pragma: no cover - corruption is rare
        logger.warning("Unable to read cached market data at %s", cache_file)
        return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            col[0] if isinstance(col, tuple) else col for col in df.columns.to_list()
        ]

    normalized = df.copy()
    normalized.columns = [str(col).lower().replace(" ", "_") for col in normalized.columns]

    expected = ["date", "close", "ticker", "ret"]
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"])

    missing = [col for col in expected if col not in normalized.columns]
    if missing:
        raise DataDownloadError(f"Market data frame missing columns: {missing}")

    return normalized[expected]


def _prepare_price_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw.empty:
        raise DataDownloadError(f"No market data returned for {ticker}.")

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance>=0.2 emits MultiIndex columns (Price, Ticker); drop the ticker level
        raw = raw.droplevel(-1, axis=1)

    normalized = raw.copy()
    normalized.columns = [str(col).lower().replace(" ", "_") for col in normalized.columns]

    close_col = "adj_close" if "adj_close" in normalized.columns else "close"
    if close_col not in normalized.columns:
        raise DataDownloadError(f"{ticker} download missing close column.")

    df = (
        normalized[[close_col]]
        .rename(columns={close_col: "close"})
        .reset_index(names="date")
        .assign(ticker=ticker)
        .sort_values("date")
    )
    df["ret"] = df["close"].pct_change()
    return _standardize_columns(df)


@rate_limited(calls_per_second=2.0)
def fetch_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Download daily data for a single ticker via yfinance.

    Returns a DataFrame with columns: date, ticker, close, ret.
    """

    if not ticker:
        raise ValueError("ticker is required")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")

    ticker = ticker.upper()
    cache_file = _cache_path(ticker, start_date, end_date)

    if use_cache and _cache_is_fresh(cache_file):
        cached = _load_cache(cache_file)
        if cached is not None:
            logger.info("Loaded %s from cache (%s rows).", ticker, len(cached))
            return _standardize_columns(cached)

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            logger.info("Fetching %s from yfinance (%s/%s).", ticker, attempt + 1, max_retries)
            raw = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
            df = _prepare_price_frame(raw, ticker)
            df.to_pickle(cache_file)
            logger.info("✓ %s: %s rows downloaded.", ticker, len(df))
            return df.copy()
        except Exception as exc:  # noqa: BLE001 - yfinance may raise many types
            last_error = exc
            if attempt >= max_retries - 1:
                break
            wait = 2**attempt
            logger.warning(
                "%s download failed on attempt %s/%s: %s. Retrying in %ss.",
                ticker,
                attempt + 1,
                max_retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise DataDownloadError(f"Failed to fetch {ticker} data.") from last_error


def fetch_bulk_data(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch data for multiple tickers with lightweight progress logging."""

    tickers = [ticker.upper() for ticker in tickers]
    if not tickers:
        raise ValueError("tickers list cannot be empty.")

    frames: list[pd.DataFrame] = []
    failures: dict[str, str] = {}
    total = len(tickers)

    for idx, ticker in enumerate(tickers, start=1):
        try:
            df = fetch_ticker_data(
                ticker,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache,
                max_retries=max_retries,
            )
            if df.empty:
                logger.warning("%s returned no rows.", ticker)
                continue
            frames.append(df)
        except DataDownloadError as exc:
            failures[ticker] = str(exc)
            logger.error("Failed to fetch %s: %s", ticker, exc)
        finally:
            logger.info("Progress: %s/%s tickers processed.", idx, total)

    if not frames:
        logger.warning("No market data fetched for requested tickers.")
        return pd.DataFrame(columns=["date", "ticker", "close", "ret"])

    combined = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    combined.reset_index(drop=True, inplace=True)

    if failures:
        logger.warning("Tickers failed (%s): %s", len(failures), ", ".join(sorted(failures)))

    return combined


def validate_data_quality(df: pd.DataFrame) -> dict[str, float]:
    """Return simple diagnostics for fetched price data."""

    if df.empty:
        summary = {
            "total_rows": 0,
            "unique_tickers": 0,
            "missing_returns": 0,
            "extreme_returns": 0,
            "coverage_pct": 0.0,
        }
        logger.warning("Market data DataFrame is empty.")
        return summary

    if "ret" not in df.columns:
        raise ValueError("DataFrame must include 'ret' column for validation.")
    if "ticker" not in df.columns:
        raise ValueError("DataFrame must include 'ticker' column for validation.")

    missing_returns = int(df["ret"].isna().sum())
    extreme_returns = int(df["ret"].abs().gt(EXTREME_RETURN_THRESHOLD).sum())
    coverage_pct = float(df["ret"].notna().mean() * 100)

    summary = {
        "total_rows": int(len(df)),
        "unique_tickers": int(df["ticker"].nunique()),
        "missing_returns": missing_returns,
        "extreme_returns": extreme_returns,
        "coverage_pct": coverage_pct,
    }
    logger.info("Market data quality metrics: %s", summary)
    return summary


if __name__ == "__main__":  # pragma: no cover
    sample_tickers = ["BAC", "JPM", "WFC"]
    start = "2023-01-01"
    end = "2023-12-31"

    prices = fetch_bulk_data(sample_tickers, start, end, use_cache=True)
    metrics = validate_data_quality(prices)
    logger.info("Fetched %s rows across %s tickers.", len(prices), metrics["unique_tickers"])
