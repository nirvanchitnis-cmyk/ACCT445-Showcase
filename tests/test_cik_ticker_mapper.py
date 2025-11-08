"""Tests for src/data/cik_ticker_mapper.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.data.cik_ticker_mapper import (
    enrich_cnoi_with_tickers,
    fetch_sec_ticker_mapping,
    get_ticker_batch,
    map_cik_to_ticker,
)
from src.utils.exceptions import DataValidationError, ExternalAPIError


class TestFetchSecTickerMapping:
    """Unit tests for SEC mapping downloads."""

    @patch("src.data.cik_ticker_mapper.requests.get")
    def test_fetch_success(self, mock_get, mock_sec_ticker_json):
        """Successful fetch returns DataFrame with expected columns."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_sec_ticker_json
        mock_get.return_value = mock_response

        df = fetch_sec_ticker_mapping()

        assert isinstance(df, pd.DataFrame)
        assert {"cik", "ticker", "title"}.issubset(df.columns)
        assert len(df) == len(mock_sec_ticker_json)
        mock_get.assert_called_once()

    @patch("src.data.cik_ticker_mapper.requests.get")
    def test_fetch_network_error(self, mock_get):
        """Network failures are surfaced as ExternalAPIError."""
        mock_get.side_effect = requests.RequestException("boom")

        with pytest.raises(ExternalAPIError):
            fetch_sec_ticker_mapping()

    @patch("src.data.cik_ticker_mapper.requests.get")
    def test_fetch_invalid_json(self, mock_get):
        """Invalid JSON response raises ExternalAPIError."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("bad json")
        mock_get.return_value = mock_response

        with pytest.raises(ExternalAPIError):
            fetch_sec_ticker_mapping()


class TestMapCikToTicker:
    """Tests for single CIK lookups."""

    def test_lookup_existing(self, mock_sec_ticker_df):
        ticker = map_cik_to_ticker(70858, mock_sec_ticker_df)
        assert ticker == "BAC"

    def test_lookup_missing(self, mock_sec_ticker_df):
        ticker = map_cik_to_ticker(1234567890, mock_sec_ticker_df)
        assert ticker is None

    def test_lookup_invalid_mapping(self):
        df = pd.DataFrame({"cik": [1]})  # Missing ticker column
        with pytest.raises(DataValidationError):
            map_cik_to_ticker(1, df)


class TestEnrichCnoiWithTickers:
    """Tests for dataframe enrichment."""

    def test_enrich_success(self, mock_sec_ticker_df):
        sample = pd.DataFrame(
            {
                "cik": [70858, 19617],
                "filing_date": pd.to_datetime(["2023-01-01", "2023-02-01"]),
                "CNOI": [10.0, 15.0],
            }
        )

        enriched = enrich_cnoi_with_tickers(sample, mapping_df=mock_sec_ticker_df)

        assert "ticker" in enriched.columns
        assert enriched["ticker"].notna().all()
        assert set(enriched["ticker"]) == {"BAC", "JPM"}

    def test_enrich_missing_cik_column(self):
        with pytest.raises(DataValidationError):
            enrich_cnoi_with_tickers(pd.DataFrame({"not_cik": []}))

    def test_enrich_with_unmapped_cik(self, mock_sec_ticker_df):
        sample = pd.DataFrame(
            {
                "cik": [99999],
                "filing_date": pd.to_datetime(["2023-01-01"]),
                "CNOI": [12.0],
            }
        )

        enriched = enrich_cnoi_with_tickers(sample, mapping_df=mock_sec_ticker_df)
        assert enriched["ticker"].isna().all()


class TestGetTickerBatch:
    """Tests for batch lookup wrapper."""

    @patch("src.data.cik_ticker_mapper.fetch_sec_ticker_mapping")
    @patch("src.data.cik_ticker_mapper.time.sleep")
    def test_batch_fetch(self, mock_sleep, mock_fetch, mock_sec_ticker_df):
        mock_sleep.return_value = None
        mock_fetch.return_value = mock_sec_ticker_df

        result = get_ticker_batch([70858, 19617])

        assert result == {70858: "BAC", 19617: "JPM"}
        assert mock_sleep.call_count == 2

    @patch("src.data.cik_ticker_mapper.fetch_sec_ticker_mapping")
    @patch("src.data.cik_ticker_mapper.time.sleep")
    def test_batch_handles_missing(self, mock_sleep, mock_fetch, mock_sec_ticker_df):
        mock_sleep.return_value = None
        mock_fetch.return_value = mock_sec_ticker_df

        result = get_ticker_batch([999999])

        assert result == {}
