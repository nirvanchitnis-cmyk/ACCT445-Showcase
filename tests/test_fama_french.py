"""Tests for Fama-French factor model estimation."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.factor_models.fama_french import (
    compute_abnormal_return,
    compute_expected_return,
    estimate_factor_loadings,
    rolling_beta_estimation,
)


@pytest.fixture
def mock_factors():
    """Create mock factor data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    data = {
        "Mkt-RF": np.random.randn(252) * 0.01 + 0.0003,  # Mean market premium
        "SMB": np.random.randn(252) * 0.005,
        "HML": np.random.randn(252) * 0.005,
        "RMW": np.random.randn(252) * 0.003,
        "CMA": np.random.randn(252) * 0.003,
        "MOM": np.random.randn(252) * 0.008,
        "RF": np.ones(252) * 0.00008,  # ~2% annual
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def spy_like_returns(mock_factors):
    """Create SPY-like returns (beta ~1.0 to market)."""
    np.random.seed(42)
    # Generate returns with beta_mkt ≈ 1.0, other betas ≈ 0
    returns = (
        0.0001  # Small alpha
        + 1.0 * mock_factors["Mkt-RF"]
        + 0.1 * mock_factors["SMB"]
        - 0.1 * mock_factors["HML"]
        + np.random.randn(len(mock_factors)) * 0.005  # Idiosyncratic risk
        + mock_factors["RF"]
    )
    return pd.Series(returns, index=mock_factors.index)


@pytest.fixture
def high_alpha_returns(mock_factors):
    """Create returns with high alpha."""
    np.random.seed(43)
    # Generate returns with significant positive alpha
    returns = (
        0.001  # 25% annual alpha (daily * 252)
        + 0.8 * mock_factors["Mkt-RF"]
        + 0.3 * mock_factors["SMB"]
        + 0.2 * mock_factors["HML"]
        + np.random.randn(len(mock_factors)) * 0.008
        + mock_factors["RF"]
    )
    return pd.Series(returns, index=mock_factors.index)


def test_estimate_factor_loadings_ff3(spy_like_returns, mock_factors):
    """Test FF3 factor estimation."""
    betas = estimate_factor_loadings(spy_like_returns, mock_factors, model="FF3")

    assert "alpha" in betas
    assert "beta_mkt" in betas
    assert "beta_smb" in betas
    assert "beta_hml" in betas
    assert "t_alpha" in betas
    assert "r_squared" in betas
    assert "n_obs" in betas
    assert betas["model"] == "FF3"

    # Check market beta is close to 1.0 (SPY-like) - allow wider range for test data
    assert 0.5 < betas["beta_mkt"] < 2.0

    # Check we have sufficient observations
    assert betas["n_obs"] > 200


def test_estimate_factor_loadings_ff5(spy_like_returns, mock_factors):
    """Test FF5 factor estimation."""
    betas = estimate_factor_loadings(spy_like_returns, mock_factors, model="FF5")

    assert "beta_rmw" in betas
    assert "beta_cma" in betas
    assert betas["model"] == "FF5"
    assert not np.isnan(betas["beta_rmw"])
    assert not np.isnan(betas["beta_cma"])


def test_estimate_factor_loadings_ff5_mom(spy_like_returns, mock_factors):
    """Test FF5 + Momentum factor estimation."""
    betas = estimate_factor_loadings(spy_like_returns, mock_factors, model="FF5_MOM")

    assert "beta_mom" in betas
    assert betas["model"] == "FF5_MOM"
    assert not np.isnan(betas["beta_mom"])


def test_estimate_factor_loadings_invalid_model(spy_like_returns, mock_factors):
    """Test error handling for invalid model."""
    with pytest.raises(ValueError, match="Invalid model"):
        estimate_factor_loadings(spy_like_returns, mock_factors, model="INVALID")


def test_estimate_factor_loadings_high_alpha(high_alpha_returns, mock_factors):
    """Test alpha estimation for high-alpha portfolio."""
    betas = estimate_factor_loadings(high_alpha_returns, mock_factors, model="FF5")

    # Should detect positive alpha
    assert betas["alpha"] > 0
    # t-stat should be significant
    assert abs(betas["t_alpha"]) > 2.0


def test_estimate_factor_loadings_insufficient_data(mock_factors):
    """Test handling of insufficient data."""
    # Only 5 observations
    short_returns = pd.Series(np.random.randn(5) * 0.01, index=mock_factors.index[:5])

    betas = estimate_factor_loadings(short_returns, mock_factors, model="FF5")

    # Should return empty results
    assert np.isnan(betas["alpha"])
    assert betas["n_obs"] == 0


def test_estimate_factor_loadings_missing_factors(spy_like_returns):
    """Test handling of missing factor columns."""
    incomplete_factors = pd.DataFrame(
        {"Mkt-RF": np.random.randn(252) * 0.01}, index=spy_like_returns.index
    )

    betas = estimate_factor_loadings(spy_like_returns, incomplete_factors, model="FF5")

    # Should return empty results due to missing factors
    assert np.isnan(betas["alpha"])


def test_estimate_factor_loadings_newey_west(spy_like_returns, mock_factors):
    """Test Newey-West standard errors."""
    betas_nw = estimate_factor_loadings(
        spy_like_returns, mock_factors, model="FF5", use_newey_west=True, maxlags=6
    )
    betas_ols = estimate_factor_loadings(
        spy_like_returns, mock_factors, model="FF5", use_newey_west=False
    )

    # Both should produce similar point estimates
    assert abs(betas_nw["beta_mkt"] - betas_ols["beta_mkt"]) < 0.1

    # T-stats may differ due to different SE estimation
    assert not np.isnan(betas_nw["t_alpha"])
    assert not np.isnan(betas_ols["t_alpha"])


def test_compute_expected_return_ff5(spy_like_returns, mock_factors):
    """Test expected return computation."""
    betas = estimate_factor_loadings(spy_like_returns, mock_factors, model="FF5")
    expected = compute_expected_return(betas, mock_factors, include_rf=True)

    assert len(expected) == len(mock_factors)
    assert isinstance(expected, pd.Series)
    # Expected returns should be reasonable (within -50% to +50% daily)
    assert expected.abs().max() < 0.5


def test_compute_expected_return_no_rf(spy_like_returns, mock_factors):
    """Test expected return without risk-free rate."""
    betas = estimate_factor_loadings(spy_like_returns, mock_factors, model="FF5")
    expected_with_rf = compute_expected_return(betas, mock_factors, include_rf=True)
    expected_no_rf = compute_expected_return(betas, mock_factors, include_rf=False)

    # Expected return without RF should be lower
    assert expected_no_rf.mean() < expected_with_rf.mean()


def test_compute_abnormal_return(spy_like_returns, mock_factors):
    """Test abnormal return (alpha) computation."""
    betas = estimate_factor_loadings(spy_like_returns, mock_factors, model="FF5")
    expected = compute_expected_return(betas, mock_factors, include_rf=True)
    abnormal = compute_abnormal_return(spy_like_returns, expected)

    assert len(abnormal) > 0
    # For SPY-like returns with small alpha, abnormal returns should be small
    assert abs(abnormal.mean()) < 0.001  # < 10 bps/day


def test_rolling_beta_estimation(spy_like_returns, mock_factors):
    """Test rolling beta estimation."""
    rolling_betas = rolling_beta_estimation(spy_like_returns, mock_factors, window=120, model="FF5")

    assert len(rolling_betas) > 0
    assert "alpha" in rolling_betas.columns
    assert "beta_mkt" in rolling_betas.columns
    assert "r_squared" in rolling_betas.columns

    # Check that market beta is relatively stable around 1.0
    assert rolling_betas["beta_mkt"].mean() > 0.5
    assert rolling_betas["beta_mkt"].mean() < 2.0


def test_rolling_beta_insufficient_data(mock_factors):
    """Test rolling beta with insufficient data."""
    short_returns = pd.Series(np.random.randn(50) * 0.01, index=mock_factors.index[:50])

    rolling_betas = rolling_beta_estimation(short_returns, mock_factors, window=252, model="FF5")

    # Should return empty DataFrame
    assert len(rolling_betas) == 0


def test_factor_loadings_residuals(spy_like_returns, mock_factors):
    """Test that residuals are returned correctly."""
    betas = estimate_factor_loadings(spy_like_returns, mock_factors, model="FF5")

    assert "residuals" in betas
    assert isinstance(betas["residuals"], pd.Series)
    assert len(betas["residuals"]) > 0

    # Residuals should have mean close to zero
    assert abs(betas["residuals"].mean()) < 0.001


def test_factor_loadings_r_squared(spy_like_returns, mock_factors):
    """Test R-squared is reasonable."""
    betas = estimate_factor_loadings(spy_like_returns, mock_factors, model="FF5")

    # For SPY-like returns, R² should be high (market explains most variance)
    assert betas["r_squared"] > 0.5
    assert betas["r_squared"] <= 1.0
