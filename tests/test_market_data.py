"""Tests for the yfinance market data utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.utils import market_data
from src.utils.exceptions import DataDownloadError


def _make_price_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    closes = [100, 101, 103, 102, 104]
    return pd.DataFrame({"Close": closes}, index=dates)


def _configure_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / "yfinance"
    cache_dir.mkdir()
    monkeypatch.setattr(market_data, "CACHE_DIR", cache_dir)
    return cache_dir


class TestFetchTickerData:
    """Unit tests for single-ticker downloads."""

    def test_download_and_cache(self, tmp_path, monkeypatch):
        cache_dir = _configure_cache(tmp_path, monkeypatch)
        frame = _make_price_frame()
        monkeypatch.setattr(market_data.yf, "download", MagicMock(return_value=frame))

        result = market_data.fetch_ticker_data("BAC", "2023-01-01", "2023-01-10", use_cache=False)

        assert {"date", "ticker", "close", "ret"} <= set(result.columns)
        assert len(result) == len(frame)
        cache_file = cache_dir / "BAC_2023-01-01_2023-01-10.pkl"
        assert cache_file.exists()

    def test_uses_cache_when_fresh(self, tmp_path, monkeypatch):
        cache_dir = _configure_cache(tmp_path, monkeypatch)
        frame = _make_price_frame()
        cache_file = cache_dir / "BAC_2023-01-01_2023-01-10.pkl"
        frame_prepared = frame.rename(columns={"Close": "close"}).reset_index(names="date")
        frame_prepared["ticker"] = "BAC"
        frame_prepared["ret"] = frame_prepared["close"].pct_change()
        frame_prepared.to_pickle(cache_file)

        # Ensure cache is treated as fresh
        monkeypatch.setattr(market_data, "CACHE_TTL_HOURS", 10_000)
        monkeypatch.setattr(market_data.yf, "download", MagicMock(side_effect=AssertionError))

        result = market_data.fetch_ticker_data("BAC", "2023-01-01", "2023-01-10", use_cache=True)
        assert len(result) == len(frame_prepared)

    def test_retries_then_raises(self, tmp_path, monkeypatch):
        _configure_cache(tmp_path, monkeypatch)
        monkeypatch.setattr(market_data.time, "sleep", lambda *_: None)
        monkeypatch.setattr(
            market_data.yf,
            "download",
            MagicMock(side_effect=ValueError("yfinance error")),
        )

        with pytest.raises(DataDownloadError):
            market_data.fetch_ticker_data(
                "BAC", "2023-01-01", "2023-01-10", use_cache=False, max_retries=2
            )


class TestFetchBulkData:
    """Tests for multi-ticker fetching."""

    def test_combines_successful_downloads(self, tmp_path, monkeypatch):
        _configure_cache(tmp_path, monkeypatch)
        good_frame = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="D"),
                "ticker": "BAC",
                "close": [100, 101, 102],
                "ret": [None, 0.01, 0.009],
            }
        )

        def fake_fetch(ticker, *_, **__):
            if ticker == "BAC":
                return good_frame
            raise DataDownloadError("boom")

        monkeypatch.setattr(market_data, "fetch_ticker_data", fake_fetch)

        result = market_data.fetch_bulk_data(["BAC", "FAIL"], "2023-01-01", "2023-01-05")
        assert len(result) == len(good_frame)
        assert result["ticker"].unique().tolist() == ["BAC"]

    def test_empty_ticker_list_raises(self):
        with pytest.raises(ValueError):
            market_data.fetch_bulk_data([], "2023-01-01", "2023-01-05")


class TestParallelTickerFetch:
    """Tests for the parallel download helper."""

    def test_parallel_fetch_handles_failures(self, tmp_path, monkeypatch):
        _configure_cache(tmp_path, monkeypatch)

        template = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=2, freq="D"),
                "ticker": ["AAA", "AAA"],
                "close": [10.0, 10.5],
                "ret": [None, 0.05],
            }
        )

        def fake_fetch(ticker, *_, **__):
            if ticker == "AAA":
                return template
            raise DataDownloadError("boom")

        monkeypatch.setattr(market_data, "fetch_ticker_data", fake_fetch)

        result = market_data.parallel_ticker_fetch(
            ["AAA", "BBB"],
            "2023-01-01",
            "2023-01-05",
            n_jobs=1,
        )
        assert result["ticker"].unique().tolist() == ["AAA"]

    def test_fetch_bulk_data_delegates_to_parallel(self, monkeypatch):
        called = {"value": False}

        def fake_parallel(*args, **kwargs):
            called["value"] = True
            return pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=1),
                    "ticker": ["AAA"],
                    "close": [1.0],
                    "ret": [0.0],
                }
            )

        monkeypatch.setattr(market_data, "parallel_ticker_fetch", fake_parallel)
        market_data.fetch_bulk_data(
            ["AAA"],
            "2023-01-01",
            "2023-01-05",
            parallel=True,
        )
        assert called["value"]


class TestValidateDataQuality:
    """Tests for validation helper."""

    def test_validation_metrics(self):
        df = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=3, freq="D"),
                "ticker": ["BAC", "BAC", "JPM"],
                "close": [100, 101, 100],
                "ret": [None, 0.02, -0.35],
            }
        )

        metrics = market_data.validate_data_quality(df)
        assert metrics["total_rows"] == 3
        assert metrics["unique_tickers"] == 2
        assert metrics["missing_returns"] == 1
        assert metrics["extreme_returns"] == 1
        assert 0 <= metrics["coverage_pct"] <= 100

    def test_empty_dataframe(self):
        metrics = market_data.validate_data_quality(
            pd.DataFrame(columns=["date", "ticker", "close", "ret"])
        )
        assert metrics["total_rows"] == 0
        assert metrics["coverage_pct"] == 0.0
