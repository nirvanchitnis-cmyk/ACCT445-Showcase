"""Pytest configuration and reusable fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_cnoi_path() -> Path:
    """Path to the bundled sample CNOI CSV."""
    return Path(__file__).parent.parent / "config" / "sample_cnoi.csv"


@pytest.fixture
def sample_cnoi_data(sample_cnoi_path: Path) -> pd.DataFrame:
    """Load the sample CNOI dataset shipped with the repo."""
    df = pd.read_csv(sample_cnoi_path, parse_dates=["filing_date"])
    if "quarter" not in df:
        df["quarter"] = df["filing_date"].dt.to_period("Q")
    return df


@pytest.fixture
def mock_sec_ticker_json() -> dict[str, dict[str, str]]:
    """Mock payload returned by the SEC ticker endpoint."""
    return {
        "0": {"cik_str": "70858", "ticker": "BAC", "title": "BANK OF AMERICA CORP /DE/"},
        "1": {"cik_str": "19617", "ticker": "JPM", "title": "JPMORGAN CHASE & CO"},
        "2": {"cik_str": "72971", "ticker": "WFC", "title": "WELLS FARGO & COMPANY"},
    }


@pytest.fixture
def mock_sec_ticker_df(mock_sec_ticker_json: dict[str, dict[str, str]]) -> pd.DataFrame:
    """Mocked SEC mapping DataFrame for offline testing."""
    records = [
        {"cik": int(payload["cik_str"]), "ticker": payload["ticker"], "title": payload["title"]}
        for payload in mock_sec_ticker_json.values()
    ]
    return pd.DataFrame(records)


@pytest.fixture
def mock_returns_data() -> pd.DataFrame:
    """Synthetic return series used across tests."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    tickers = ["BAC", "JPM"]

    rows = []
    for ticker in tickers:
        base = 0.001 if ticker == "BAC" else 0.0005
        noise = np.random.normal(scale=0.015, size=len(dates))
        rows.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "return": base + noise,
                }
            )
        )

    df = pd.concat(rows, ignore_index=True)
    df["ret_fwd"] = df.groupby("ticker")["return"].shift(-1)
    return df.dropna(subset=["return"])


@pytest.fixture
def sample_decile_data() -> pd.DataFrame:
    """Sample dataset for decile backtest tests."""
    np.random.seed(123)
    dates = pd.date_range("2022-01-01", periods=12, freq="QE-DEC")
    tickers = [f"BANK{i:02d}" for i in range(30)]

    rows = []
    for date in dates:
        for ticker in tickers:
            cnoi = np.random.uniform(5, 35)
            ret = np.random.normal(0.02 - 0.0005 * cnoi, 0.05)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "CNOI": cnoi,
                    "ret_fwd": ret,
                    "market_cap": np.random.uniform(1e9, 5e10),
                }
            )

    df = pd.DataFrame(rows)
    return df


@pytest.fixture
def sample_event_study_returns() -> pd.DataFrame:
    """Synthetic stock return panel for event study tests."""
    np.random.seed(7)
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    tickers = ["BAC", "JPM", "WFC", "USB"]
    rows = []
    for ticker in tickers:
        beta = 1.2 if ticker in {"BAC", "JPM"} else 0.9
        alpha = np.random.uniform(-0.0002, 0.0002)
        market = np.random.normal(0.0005, 0.01, len(dates))
        returns = alpha + beta * market + np.random.normal(0, 0.01, len(dates))
        rows.append(pd.DataFrame({"date": dates, "ticker": ticker, "return": returns}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def sample_market_returns(sample_event_study_returns: pd.DataFrame) -> pd.Series:
    """Synthetic market factor aligned with the event study data."""
    daily = sample_event_study_returns.groupby("date")["return"].mean()
    daily.name = "mkt_ret"
    return daily


@pytest.fixture
def sample_panel_data() -> pd.DataFrame:
    """Synthetic panel dataset for regression tests."""
    np.random.seed(321)
    tickers = [f"BANK{i:02d}" for i in range(12)]
    quarters = pd.period_range("2020Q1", periods=10, freq="Q")

    rows = []
    for ticker in tickers:
        entity_alpha = np.random.normal(0, 0.005)
        for quarter in quarters:
            time_beta = np.random.normal(0, 0.003)
            cnoi = np.random.uniform(5, 30)
            log_mcap = np.random.uniform(20, 26)
            ret = (
                entity_alpha
                + time_beta
                - 0.002 * cnoi
                + 0.0005 * log_mcap
                + np.random.normal(0, 0.01)
            )
            rows.append(
                {
                    "ticker": ticker,
                    "quarter": quarter,
                    "CNOI": cnoi,
                    "log_mcap": log_mcap,
                    "ret_fwd": ret,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def dimension_data() -> pd.DataFrame:
    """Synthetic dataset with all seven CNOI dimensions for dimension analysis tests."""
    np.random.seed(42)
    tickers = [f"STOCK{i:02d}" for i in range(30)]
    quarters = pd.period_range("2020Q1", periods=10, freq="Q")
    dim_keys = list("DGRJTSX")

    rows = []
    for quarter in quarters:
        for ticker in tickers:
            dims = {dim: np.random.uniform(5, 15) for dim in dim_keys}
            ret_fwd = -0.004 * dims["S"] - 0.003 * dims["R"] + np.random.normal(0, 0.01)
            rows.append(
                {
                    "ticker": ticker,
                    "quarter": quarter,
                    "date": quarter.to_timestamp(),
                    **dims,
                    "CNOI": sum(dims.values()),
                    "ret_fwd": ret_fwd,
                    "market_cap": np.random.uniform(5e8, 5e10),
                }
            )

    return pd.DataFrame(rows)
