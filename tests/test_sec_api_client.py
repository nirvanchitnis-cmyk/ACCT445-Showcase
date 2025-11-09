"""Tests for the SEC API client."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import src.data.sec_api_client as sec_api_client
from src.utils.exceptions import ExternalAPIError


def _setup_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    cache_dir = tmp_path / "sec"
    cache_dir.mkdir()
    monkeypatch.setattr(sec_api_client, "CACHE_DIR", cache_dir)
    return cache_dir


class TestSecApiClient:
    """Verify retry logic and caching for SEC API client."""

    def test_fetch_success(self, mock_sec_ticker_json, tmp_path, monkeypatch):
        cache_dir = _setup_cache(monkeypatch, tmp_path)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_sec_ticker_json
        mock_response.raise_for_status.return_value = None

        with patch("src.data.sec_api_client.requests.get", return_value=mock_response) as mock_get:
            result = sec_api_client.fetch_sec_ticker_mapping(use_cache=False)

        assert "0000070858" in result
        assert result["0000070858"]["ticker"] == "BAC"
        mock_get.assert_called_once()
        assert (cache_dir / "company_tickers.json").exists()

    def test_retry_on_403(self, mock_sec_ticker_json, tmp_path, monkeypatch):
        _setup_cache(monkeypatch, tmp_path)

        forbidden = MagicMock()
        forbidden.status_code = 403
        forbidden.raise_for_status.side_effect = requests.exceptions.HTTPError(response=forbidden)

        success = MagicMock()
        success.status_code = 200
        success.raise_for_status.return_value = None
        success.json.return_value = mock_sec_ticker_json

        side_effects = [forbidden, success]

        with (
            patch("src.data.sec_api_client.time.sleep") as mock_sleep,
            patch("src.data.sec_api_client.requests.get", side_effect=side_effects) as mock_get,
        ):
            result = sec_api_client.fetch_sec_ticker_mapping(use_cache=False, max_retries=2)

        assert result["0000070858"]["ticker"] == "BAC"
        assert mock_get.call_count == 2
        mock_sleep.assert_called_once()

    def test_exhausted_retries_raise(self, tmp_path, monkeypatch):
        _setup_cache(monkeypatch, tmp_path)

        forbidden = MagicMock()
        forbidden.status_code = 403
        forbidden.raise_for_status.side_effect = requests.exceptions.HTTPError(response=forbidden)

        with (
            patch("src.data.sec_api_client.time.sleep") as mock_sleep,
            patch("src.data.sec_api_client.requests.get", return_value=forbidden),
        ):
            with pytest.raises(ExternalAPIError):
                sec_api_client.fetch_sec_ticker_mapping(use_cache=False, max_retries=2)

        assert mock_sleep.call_count == 1

    def test_cache_usage(self, tmp_path, monkeypatch):
        cache_dir = _setup_cache(monkeypatch, tmp_path)
        cache_file = cache_dir / "company_tickers.json"
        cached = {"0000000001": {"ticker": "ABC", "title": "Alpha Beta Co"}}
        cache_file.write_text(json.dumps(cached), encoding="utf-8")

        with patch("src.data.sec_api_client.requests.get") as mock_get:
            result = sec_api_client.fetch_sec_ticker_mapping(use_cache=True)

        assert result == cached
        mock_get.assert_not_called()
