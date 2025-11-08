"""Tests for src/utils/data_loader.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.utils.data_loader import (
    compute_forward_returns,
    create_sample_cnoi_file,
    load_cnoi_data,
    load_market_returns,
    merge_cnoi_with_returns,
)
from src.utils.exceptions import DataDownloadError, DataValidationError


class TestLoadCnoiData:
    """Tests for CSV ingestion."""

    def test_load_sample_dataset(self, sample_cnoi_path: Path):
        df = load_cnoi_data(str(sample_cnoi_path))
        assert not df.empty
        assert pd.api.types.is_datetime64_any_dtype(df["filing_date"])
        assert "quarter" in df.columns

    def test_missing_columns_raise(self, tmp_path: Path):
        invalid = tmp_path / "invalid.csv"
        pd.DataFrame({"cik": [1], "filing_date": ["2023-01-01"]}).to_csv(invalid, index=False)

        with pytest.raises(DataValidationError):
            load_cnoi_data(str(invalid))


class TestLoadMarketReturns:
    """Tests for yfinance wrapper."""

    @patch("src.utils.data_loader.yf.download")
    def test_single_ticker_download(self, mock_download):
        idx = pd.date_range("2023-01-01", periods=4, freq="D")
        mock_download.return_value = pd.DataFrame(
            {
                "Adj Close": [100, 102, 101, 105],
                "Volume": [1_000_000, 1_100_000, 900_000, 950_000],
            },
            index=idx,
        )

        df = load_market_returns(["AAPL"], "2023-01-01", "2023-01-05")

        assert set(df["ticker"]) == {"AAPL"}
        assert {"date", "return", "price", "volume"}.issubset(df.columns)
        assert len(df) == len(idx) - 1  # drop first row due to pct_change

    @patch("src.utils.data_loader.yf.download")
    def test_multi_ticker_download(self, mock_download):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        mock_download.return_value = pd.DataFrame(
            {
                ("Adj Close", "AAPL"): [100, 102, 101],
                ("Volume", "AAPL"): [1, 2, 3],
                ("Adj Close", "MSFT"): [200, 204, 202],
                ("Volume", "MSFT"): [4, 5, 6],
            },
            index=idx,
        )

        df = load_market_returns(["AAPL", "MSFT"], "2023-01-01", "2023-01-04")

        assert set(df["ticker"]) == {"AAPL", "MSFT"}
        assert not df.empty

    @patch("src.utils.data_loader.yf.download")
    def test_empty_download_raises(self, mock_download):
        mock_download.return_value = pd.DataFrame()

        with pytest.raises(DataDownloadError):
            load_market_returns(["AAPL"], "2023-01-01", "2023-01-05")

    def test_invalid_date_range(self):
        with pytest.raises(DataValidationError):
            load_market_returns(["AAPL"], "2023-02-01", "2023-01-01")

    def test_empty_ticker_list(self):
        with pytest.raises(DataValidationError):
            load_market_returns([], "2023-01-01", "2023-01-05")

    @patch("src.utils.data_loader.yf.download")
    def test_missing_ticker_in_output(self, mock_download):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        mock_download.return_value = pd.DataFrame(
            {
                ("Adj Close", "AAPL"): [100, 102, 101],
                ("Volume", "AAPL"): [1, 2, 3],
            },
            index=idx,
        )

        df = load_market_returns(["AAPL", "MSFT"], "2023-01-01", "2023-01-04")
        assert set(df["ticker"]) == {"AAPL"}

    @patch("src.utils.data_loader.yf.download")
    def test_missing_price_column_triggers_error(self, mock_download):
        idx = pd.date_range("2023-01-01", periods=3, freq="D")
        mock_download.return_value = pd.DataFrame(
            {
                ("Volume", "AAPL"): [1, 2, 3],
            },
            index=idx,
        )

        with pytest.raises(DataDownloadError):
            load_market_returns(["AAPL"], "2023-01-01", "2023-01-04")

    @patch("src.utils.data_loader.yf.download")
    def test_weekly_frequency_resample(self, mock_download):
        idx = pd.date_range("2023-01-01", periods=10, freq="D")
        mock_download.return_value = pd.DataFrame(
            {
                "Adj Close": np.linspace(100, 110, 10),
                "Volume": np.arange(10) + 1,
            },
            index=idx,
        )

        df = load_market_returns(["AAPL"], "2023-01-01", "2023-01-11", frequency="weekly")
        assert df["date"].dt.weekday.eq(6).all()


class TestComputeForwardReturns:
    """Tests for forward return computations."""

    def test_quarterly_forward_returns(self, mock_returns_data: pd.DataFrame):
        result = compute_forward_returns(mock_returns_data, horizon=1, frequency="daily")
        assert "ret_fwd" in result.columns
        assert result["ret_fwd"].isna().sum() > 0  # trailing rows should be NaN

    def test_invalid_horizon(self, mock_returns_data: pd.DataFrame):
        with pytest.raises(DataValidationError):
            compute_forward_returns(mock_returns_data, horizon=0)

    def test_quarterly_frequency(self, mock_returns_data: pd.DataFrame):
        result = compute_forward_returns(mock_returns_data, horizon=1, frequency="quarterly")
        assert "ret_fwd" in result.columns
        assert result["ret_fwd"].notna().any()


class TestMergeCnoiWithReturns:
    """Tests for merge logic."""

    def test_merge_success(self, mock_returns_data: pd.DataFrame):
        cnoi = pd.DataFrame(
            {
                "cik": [1, 2],
                "ticker": ["BAC", "JPM"],
                "filing_date": pd.to_datetime(["2023-01-01", "2023-01-05"]),
                "CNOI": [12.0, 15.0],
            }
        )

        merged = merge_cnoi_with_returns(cnoi, mock_returns_data, lag_days=2)

        assert {"ticker", "CNOI", "return"}.issubset(merged.columns)
        assert (merged["date"] >= merged["filing_date"]).all()

    def test_missing_columns_raise(self, mock_returns_data: pd.DataFrame):
        with pytest.raises(DataValidationError):
            merge_cnoi_with_returns(pd.DataFrame({"bad": []}), mock_returns_data)


class TestCreateSampleCnoiFile:
    """Tests for sample file creation helper."""

    def test_sample_file_created(self, tmp_path: Path):
        full = tmp_path / "full.csv"
        rows = []
        for cik in range(5):
            for month in range(3):
                rows.append(
                    {
                        "cik": cik,
                        "filing_date": f"2023-0{month+1}-15",
                        "CNOI": np.random.uniform(5, 30),
                    }
                )
        pd.DataFrame(rows).to_csv(full, index=False)

        output = tmp_path / "sample.csv"
        create_sample_cnoi_file(str(full), str(output), n_top=2, n_bottom=2)

        assert output.exists()
        sample = pd.read_csv(output)
        assert not sample.empty
