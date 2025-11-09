"""Tests for factor data fetching and caching."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.utils.factor_data import (
    fetch_all_factors,
    fetch_fama_french_factors,
    load_factors_from_csv,
    save_factors_to_dvc,
)


@pytest.fixture
def mock_ff5_data():
    """Mock Fama-French 5-factor data."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    data = {
        "Mkt-RF": np.random.randn(10) * 0.01,
        "SMB": np.random.randn(10) * 0.005,
        "HML": np.random.randn(10) * 0.005,
        "RMW": np.random.randn(10) * 0.003,
        "CMA": np.random.randn(10) * 0.003,
        "RF": np.random.rand(10) * 0.0001,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def mock_momentum_data():
    """Mock momentum factor data."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    data = {"MOM": np.random.randn(10) * 0.008}
    return pd.DataFrame(data, index=dates)


def test_fetch_fama_french_factors_ff5(mock_ff5_data):
    """Test fetching FF5 factors."""
    with patch("src.utils.factor_data.web.DataReader") as mock_reader:
        # Mock returns dict with key 0 containing data (in percentages)
        mock_reader.return_value = {0: mock_ff5_data * 100}

        factors = fetch_fama_french_factors(
            "F-F_Research_Data_5_Factors_2x3_daily",
            start_date="2023-01-01",
            end_date="2023-01-31",
            disable_cache=True,
        )

        assert len(factors) == 10
        assert "Mkt-RF" in factors.columns
        assert "SMB" in factors.columns
        assert "HML" in factors.columns
        assert "RMW" in factors.columns
        assert "CMA" in factors.columns
        assert "RF" in factors.columns
        assert isinstance(factors.index, pd.DatetimeIndex)

        # Check that percentages were converted to decimals
        assert factors["Mkt-RF"].abs().max() < 1.0  # Should be < 100% as decimals


def test_fetch_fama_french_factors_momentum(mock_momentum_data):
    """Test fetching momentum factor."""
    with patch("src.utils.factor_data.web.DataReader") as mock_reader:
        mock_reader.return_value = {0: mock_momentum_data * 100}

        factors = fetch_fama_french_factors(
            "F-F_Momentum_Factor_daily",
            start_date="2023-01-01",
            end_date="2023-01-31",
            disable_cache=True,
        )

        assert len(factors) == 10
        assert "MOM" in factors.columns
        assert isinstance(factors.index, pd.DatetimeIndex)


def test_fetch_fama_french_factors_cache_hit(mock_ff5_data, tmp_path):
    """Test that cache is respected when disable_cache is used."""
    with patch("src.utils.factor_data.web.DataReader") as mock_reader:
        mock_reader.return_value = {0: mock_ff5_data * 100}

        # First call with cache disabled - should fetch
        factors1 = fetch_fama_french_factors(
            "F-F_Research_Data_5_Factors_2x3_daily",
            start_date="2023-01-01",
            end_date="2023-01-31",
            disable_cache=True,
        )

        # Should have been called once
        assert mock_reader.call_count >= 1
        assert len(factors1) == 10


def test_fetch_all_factors(mock_ff5_data, mock_momentum_data):
    """Test fetching all factors (FF5 + Momentum)."""
    with patch("src.utils.factor_data.web.DataReader") as mock_reader:

        def side_effect(name, *args, **kwargs):
            if "5_Factors" in name:
                return {0: mock_ff5_data * 100}
            elif "Momentum" in name:
                return {0: mock_momentum_data * 100}
            return None

        mock_reader.side_effect = side_effect

        all_factors = fetch_all_factors(
            start_date="2023-01-01", end_date="2023-01-31", disable_cache=True
        )

        assert len(all_factors) == 10
        assert "Mkt-RF" in all_factors.columns
        assert "SMB" in all_factors.columns
        assert "HML" in all_factors.columns
        assert "RMW" in all_factors.columns
        assert "CMA" in all_factors.columns
        assert "MOM" in all_factors.columns
        assert "RF" in all_factors.columns


def test_save_factors_to_dvc(mock_ff5_data, tmp_path):
    """Test saving factors to CSV and DVC."""
    output_path = tmp_path / "ff5_test.csv"

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        save_factors_to_dvc(mock_ff5_data, output_path)

        # Check CSV was created
        assert output_path.exists()

        # Check DVC was called
        mock_run.assert_called_once()
        assert "dvc" in mock_run.call_args[0][0]
        assert "add" in mock_run.call_args[0][0]


def test_save_factors_to_dvc_no_dvc(mock_ff5_data, tmp_path):
    """Test saving factors when DVC is not installed."""
    output_path = tmp_path / "ff5_test.csv"

    with patch("subprocess.run", side_effect=FileNotFoundError):
        # Should not raise error, just skip DVC
        save_factors_to_dvc(mock_ff5_data, output_path)

        # Check CSV was still created
        assert output_path.exists()


def test_load_factors_from_csv(mock_ff5_data, tmp_path):
    """Test loading factors from CSV."""
    csv_path = tmp_path / "factors.csv"
    mock_ff5_data.to_csv(csv_path)

    loaded = load_factors_from_csv(csv_path)

    assert len(loaded) == len(mock_ff5_data)
    assert list(loaded.columns) == list(mock_ff5_data.columns)
    assert isinstance(loaded.index, pd.DatetimeIndex)


def test_load_factors_from_csv_not_found(tmp_path):
    """Test loading factors from non-existent CSV."""
    csv_path = tmp_path / "nonexistent.csv"

    with pytest.raises(FileNotFoundError):
        load_factors_from_csv(csv_path)


def test_fetch_fama_french_factors_error():
    """Test error handling when fetch fails."""
    with patch("src.utils.factor_data.web.DataReader", side_effect=Exception("Network error")):
        with pytest.raises(Exception):
            fetch_fama_french_factors(
                "F-F_Research_Data_5_Factors_2x3_daily",
                start_date="2023-01-01",
                end_date="2023-01-31",
                disable_cache=True,
            )


def test_fetch_factors_date_alignment(mock_ff5_data):
    """Test that factors are properly aligned to business days."""
    with patch("src.utils.factor_data.web.DataReader") as mock_reader:
        mock_reader.return_value = {0: mock_ff5_data * 100}

        factors = fetch_fama_french_factors(
            "F-F_Research_Data_5_Factors_2x3_daily",
            start_date="2023-01-01",
            end_date="2023-01-31",
            disable_cache=True,
        )

        # Check index is sorted
        assert factors.index.is_monotonic_increasing

        # Check index is DatetimeIndex
        assert isinstance(factors.index, pd.DatetimeIndex)
