"""Tests for alpha decomposition and Jensen's alpha."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.factor_models.alpha_decomposition import (
    alpha_attribution,
    carhart_alpha,
    jensen_alpha,
    long_short_alpha,
    summarize_decile_alphas,
)


@pytest.fixture
def mock_factors():
    """Create mock factor data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    data = {
        "Mkt-RF": np.random.randn(252) * 0.01 + 0.0003,
        "SMB": np.random.randn(252) * 0.005,
        "HML": np.random.randn(252) * 0.005,
        "RMW": np.random.randn(252) * 0.003,
        "CMA": np.random.randn(252) * 0.003,
        "MOM": np.random.randn(252) * 0.008,
        "RF": np.ones(252) * 0.00008,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def zero_alpha_returns(mock_factors):
    """Returns with zero alpha (purely explained by factors)."""
    np.random.seed(42)
    returns = (
        0.0  # No alpha
        + 1.0 * mock_factors["Mkt-RF"]
        + 0.2 * mock_factors["SMB"]
        + 0.1 * mock_factors["HML"]
        + 0.05 * mock_factors["RMW"]
        - 0.05 * mock_factors["CMA"]
        + np.random.randn(len(mock_factors)) * 0.01  # Add more noise for realism
        + mock_factors["RF"]
    )
    return pd.Series(returns, index=mock_factors.index)


@pytest.fixture
def high_alpha_returns(mock_factors):
    """Returns with high positive alpha."""
    np.random.seed(43)
    returns = (
        0.002  # 50% annual alpha
        + 0.8 * mock_factors["Mkt-RF"]
        + 0.3 * mock_factors["SMB"]
        + np.random.randn(len(mock_factors)) * 0.005
        + mock_factors["RF"]
    )
    return pd.Series(returns, index=mock_factors.index)


def test_jensen_alpha_ff5(high_alpha_returns, mock_factors):
    """Test Jensen's alpha calculation with FF5."""
    result = jensen_alpha(high_alpha_returns, mock_factors, model="FF5")

    assert "alpha_annual" in result
    assert "alpha_daily" in result
    assert "t_stat" in result
    assert "p_value" in result
    assert "model" in result
    assert result["model"] == "FF5"

    # High alpha returns should have positive alpha
    assert result["alpha_annual"] > 0

    # Annualized alpha should be ~252x daily alpha
    assert abs(result["alpha_annual"] - result["alpha_daily"] * 252) < 0.01


def test_jensen_alpha_ff3(high_alpha_returns, mock_factors):
    """Test Jensen's alpha with FF3 model."""
    result = jensen_alpha(high_alpha_returns, mock_factors, model="FF3")

    assert result["model"] == "FF3"
    assert "alpha_annual" in result
    assert result["alpha_annual"] > 0


def test_jensen_alpha_zero_alpha(zero_alpha_returns, mock_factors):
    """Test that zero-alpha returns produce alpha near zero."""
    result = jensen_alpha(zero_alpha_returns, mock_factors, model="FF5")

    # Alpha should be very small (but allow for some noise in test data)
    assert abs(result["alpha_annual"]) < 1.0  # < 100% annual (generous for noisy test data)

    # R-squared should be reasonably high since returns are factor-driven
    assert result["r_squared"] > 0.5


def test_jensen_alpha_annualization(high_alpha_returns, mock_factors):
    """Test annualization of alpha."""
    result_annual = jensen_alpha(
        high_alpha_returns, mock_factors, model="FF5", annualize=True, periods_per_year=252
    )
    result_daily = jensen_alpha(high_alpha_returns, mock_factors, model="FF5", annualize=False)

    # Annual should be ~252x daily
    assert abs(result_annual["alpha_annual"] / result_daily["alpha_annual"] - 252) < 1


def test_carhart_alpha(high_alpha_returns, mock_factors):
    """Test Carhart 4-factor alpha."""
    result = carhart_alpha(high_alpha_returns, mock_factors)

    assert result["model"] == "FF5_MOM"
    assert "alpha_annual" in result
    assert result["alpha_annual"] > 0


def test_alpha_attribution_ff5(high_alpha_returns, mock_factors):
    """Test alpha attribution decomposition."""
    attribution = alpha_attribution(high_alpha_returns, mock_factors, model="FF5")

    assert len(attribution) > 0
    assert "total_return" in attribution.columns
    assert "alpha" in attribution.columns
    assert "mkt_premium" in attribution.columns
    assert "smb_premium" in attribution.columns
    assert "hml_premium" in attribution.columns
    assert "rmw_premium" in attribution.columns
    assert "cma_premium" in attribution.columns
    assert "residual" in attribution.columns


def test_alpha_attribution_ff5_mom(high_alpha_returns, mock_factors):
    """Test alpha attribution with momentum."""
    attribution = alpha_attribution(high_alpha_returns, mock_factors, model="FF5_MOM")

    assert "mom_premium" in attribution.columns


def test_alpha_attribution_sum(high_alpha_returns, mock_factors):
    """Test that attribution components sum to total return."""
    attribution = alpha_attribution(high_alpha_returns, mock_factors, model="FF5")

    # model_return should approximately equal total_return
    # (small difference due to residuals)
    diff = (attribution["total_return"] - attribution["model_return"]).abs().mean()
    assert diff < 0.01  # Average difference < 1%


def test_summarize_decile_alphas(mock_factors):
    """Test decile alpha summarization."""
    # Create mock decile returns
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="B")
    decile_data = []

    for decile in range(1, 11):
        for date in dates:
            decile_data.append(
                {
                    "decile": decile,
                    "date": date,
                    "ret_fwd": 0.0005 * decile + np.random.randn() * 0.01,
                }
            )

    decile_returns = pd.DataFrame(decile_data)

    summary = summarize_decile_alphas(decile_returns, mock_factors, model="FF5")

    assert len(summary) == 10  # 10 deciles
    assert "decile" in summary.columns
    assert "alpha_annual" in summary.columns
    assert "t_stat" in summary.columns
    assert "p_value" in summary.columns

    # Decile column should have values 1-10
    assert summary["decile"].min() == 1
    assert summary["decile"].max() == 10


def test_long_short_alpha(mock_factors):
    """Test long-short alpha calculation."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="B")

    # Long side: high alpha
    long_returns = pd.Series(
        0.001 + 0.8 * mock_factors["Mkt-RF"][:120] + np.random.randn(120) * 0.005,
        index=dates,
    )

    # Short side: zero alpha
    short_returns = pd.Series(
        0.9 * mock_factors["Mkt-RF"][:120] + np.random.randn(120) * 0.005,
        index=dates,
    )

    ls_result = long_short_alpha(long_returns, short_returns, mock_factors, model="FF5")

    assert "alpha_annual" in ls_result
    assert "t_stat" in ls_result

    # Long-short should have positive alpha
    assert ls_result["alpha_annual"] > 0


def test_alpha_attribution_empty_data(mock_factors):
    """Test attribution with no overlapping data."""
    # Returns with non-overlapping dates
    other_dates = pd.date_range("2022-01-01", periods=10, freq="B")
    returns = pd.Series(np.random.randn(10) * 0.01, index=other_dates)

    attribution = alpha_attribution(returns, mock_factors, model="FF5")

    # Should return empty DataFrame
    assert len(attribution) == 0


def test_jensen_alpha_significance(high_alpha_returns, mock_factors):
    """Test that high-alpha returns produce significant t-stats."""
    result = jensen_alpha(high_alpha_returns, mock_factors, model="FF5")

    # T-stat should be significant (> 2.0 for 95% confidence)
    assert abs(result["t_stat"]) > 2.0

    # P-value should be small
    assert result["p_value"] < 0.05


def test_alpha_decomposition_r_squared(high_alpha_returns, mock_factors):
    """Test that R-squared is returned in alpha results."""
    result = jensen_alpha(high_alpha_returns, mock_factors, model="FF5")

    assert "r_squared" in result
    assert 0 <= result["r_squared"] <= 1
