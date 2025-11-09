"""
Tests for robust alpha estimation module.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.factor_models.alpha_robust import (
    compute_alpha_with_dsr,
    deflated_sharpe_ratio,
    rolling_alpha,
)


class TestDeflatedSharpeRatio:
    """Tests for deflated_sharpe_ratio()."""

    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        result = deflated_sharpe_ratio(
            sharpe_ratio=0.8,
            n_observations=240,
            skewness=0,
            kurtosis=3,
            n_trials=1,
        )

        assert isinstance(result, dict)

    def test_contains_required_keys(self):
        """Test that result contains required keys."""
        result = deflated_sharpe_ratio(sharpe_ratio=0.8, n_observations=240, n_trials=1)

        required_keys = [
            "dsr",
            "psr",
            "adj_alpha",
            "z_stat",
            "threshold_sr",
            "is_significant",
        ]
        for key in required_keys:
            assert key in result

    def test_high_sharpe_with_one_trial(self):
        """Test that high SR with 1 trial is significant."""
        result = deflated_sharpe_ratio(
            sharpe_ratio=1.2,  # High SR
            n_observations=240,
            skewness=0,
            kurtosis=3,
            n_trials=1,  # No multiple testing
        )

        # Should be significant
        assert result["is_significant"]
        assert result["psr"] > 0.95
        assert result["dsr"] > 1.0

    def test_low_sharpe_not_significant(self):
        """Test that low SR with few observations is not significant."""
        result = deflated_sharpe_ratio(
            sharpe_ratio=0.05,  # Very low SR
            n_observations=20,  # Few observations
            n_trials=100,  # Many trials → very high threshold
        )

        # Should not be significant (very low SR + few obs + many trials)
        assert not result["is_significant"]
        assert result["dsr"] < 1.0

    def test_multiple_trials_increases_threshold(self):
        """Test that more trials increase the significance threshold."""
        sr = 0.8
        n_obs = 240

        result_1_trial = deflated_sharpe_ratio(sr, n_obs, n_trials=1)
        result_10_trials = deflated_sharpe_ratio(sr, n_obs, n_trials=10)

        # More trials → higher threshold
        assert result_10_trials["threshold_sr"] > result_1_trial["threshold_sr"]
        assert result_10_trials["dsr"] < result_1_trial["dsr"]

    def test_negative_sharpe_ratio(self):
        """Test handling of negative Sharpe ratios."""
        result = deflated_sharpe_ratio(
            sharpe_ratio=-0.5,
            n_observations=240,
            n_trials=1,
        )

        # Should have negative DSR
        assert result["dsr"] < 0
        assert not result["is_significant"]

    def test_skewness_kurtosis_effect(self):
        """Test that non-normality affects DSR."""
        sr = 0.8
        n_obs = 240

        # Normal distribution
        result_normal = deflated_sharpe_ratio(sr, n_obs, skewness=0, kurtosis=3)

        # Fat tails (high kurtosis)
        result_fat_tails = deflated_sharpe_ratio(sr, n_obs, skewness=0, kurtosis=6)

        # Fat tails increase variance → lower DSR
        assert result_fat_tails["dsr"] < result_normal["dsr"]

    def test_more_observations_increases_significance(self):
        """Test that more observations increase statistical power."""
        sr = 0.6

        result_60 = deflated_sharpe_ratio(sr, n_observations=60)
        result_240 = deflated_sharpe_ratio(sr, n_observations=240)

        # More observations → higher DSR (lower threshold)
        assert result_240["dsr"] > result_60["dsr"]

    def test_raises_on_too_few_observations(self):
        """Test raises error if n_observations <= 2."""
        with pytest.raises(ValueError, match="Need at least 3 observations"):
            deflated_sharpe_ratio(sharpe_ratio=0.8, n_observations=2)


class TestRollingAlpha:
    """Tests for rolling_alpha()."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic returns and factors."""
        np.random.seed(42)
        n_periods = 120  # ~2 years of weekly data
        dates = pd.date_range("2023-01-01", periods=n_periods, freq="W")

        # Factors (use uppercase MOM to match Ken French naming)
        factors = pd.DataFrame(
            {
                "Mkt-RF": np.random.normal(0.002, 0.02, n_periods),
                "SMB": np.random.normal(0.0, 0.01, n_periods),
                "HML": np.random.normal(0.0, 0.01, n_periods),
                "RMW": np.random.normal(0.0, 0.01, n_periods),
                "CMA": np.random.normal(0.0, 0.01, n_periods),
                "MOM": np.random.normal(0.001, 0.015, n_periods),
                "RF": np.random.normal(0.0001, 0.0001, n_periods),
            },
            index=dates,
        )

        # Returns with positive alpha
        alpha_true = 0.003  # 0.3% weekly alpha
        beta_mkt = 0.8
        returns = alpha_true + beta_mkt * factors["Mkt-RF"] + np.random.normal(0, 0.01, n_periods)
        returns = pd.Series(returns, index=dates)

        return returns, factors

    def test_returns_dataframe(self, synthetic_data):
        """Test that function returns a DataFrame."""
        returns, factors = synthetic_data

        result = rolling_alpha(returns, factors, model="FF5", window=52, step=4)

        assert isinstance(result, pd.DataFrame)

    def test_contains_required_columns(self, synthetic_data):
        """Test that result contains required columns."""
        returns, factors = synthetic_data

        result = rolling_alpha(returns, factors, model="FF5", window=52, step=4)

        required_cols = ["alpha", "t_alpha", "r_squared", "n_obs"]
        for col in required_cols:
            assert col in result.columns

    def test_window_count(self, synthetic_data):
        """Test that number of windows is correct."""
        returns, factors = synthetic_data

        window = 52
        step = 4
        n_periods = len(returns)

        result = rolling_alpha(returns, factors, window=window, step=step)

        # Expected windows: (n - window) / step + 1
        expected_windows = (n_periods - window) // step + 1
        assert len(result) == expected_windows

    def test_positive_alpha_recovered(self, synthetic_data):
        """Test that positive alpha is recovered in rolling windows."""
        returns, factors = synthetic_data

        result = rolling_alpha(returns, factors, model="FF5", window=52, step=10)

        # Most windows should have positive alpha (true alpha = 0.3% weekly)
        pct_positive = (result["alpha"] > 0).mean()
        assert pct_positive > 0.5  # At least 50% windows positive

        # Median should be close to true alpha (0.003)
        median_alpha = result["alpha"].median()
        assert 0 < median_alpha < 0.01  # Reasonable range

    def test_ff3_model(self, synthetic_data):
        """Test that FF3 model works."""
        returns, factors = synthetic_data

        result = rolling_alpha(returns, factors, model="FF3", window=52)

        assert len(result) > 0
        assert "alpha" in result.columns

    def test_carhart_model(self, synthetic_data):
        """Test that Carhart model works."""
        returns, factors = synthetic_data

        result = rolling_alpha(returns, factors, model="Carhart", window=52)

        assert len(result) > 0
        assert "alpha" in result.columns

    def test_raises_on_unknown_model(self, synthetic_data):
        """Test raises error on unknown model."""
        returns, factors = synthetic_data

        with pytest.raises(ValueError, match="Unknown model"):
            rolling_alpha(returns, factors, model="UnknownModel")

    def test_raises_on_insufficient_data(self, synthetic_data):
        """Test raises error if not enough data."""
        returns, factors = synthetic_data

        # Request window larger than data
        with pytest.raises(ValueError, match="Not enough overlapping data"):
            rolling_alpha(returns, factors, window=200)


class TestComputeAlphaWithDSR:
    """Tests for compute_alpha_with_dsr()."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic returns and factors."""
        np.random.seed(42)
        n_periods = 240  # ~1 year of daily data
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="D")

        # Factors (use uppercase MOM to match Ken French naming)
        factors = pd.DataFrame(
            {
                "Mkt-RF": np.random.normal(0.0005, 0.01, n_periods),
                "SMB": np.random.normal(0.0, 0.005, n_periods),
                "HML": np.random.normal(0.0, 0.005, n_periods),
                "RMW": np.random.normal(0.0, 0.005, n_periods),
                "CMA": np.random.normal(0.0, 0.005, n_periods),
                "MOM": np.random.normal(0.0002, 0.007, n_periods),
                "RF": np.random.normal(0.00002, 0.00001, n_periods),
            },
            index=dates,
        )

        # Returns with strong alpha
        alpha_true = 0.001  # 0.1% daily alpha
        beta_mkt = 0.9
        returns = alpha_true + beta_mkt * factors["Mkt-RF"] + np.random.normal(0, 0.008, n_periods)
        returns = pd.Series(returns, index=dates)

        return returns, factors

    def test_returns_dict(self, synthetic_data):
        """Test that function returns a dictionary."""
        returns, factors = synthetic_data

        result = compute_alpha_with_dsr(returns, factors, n_trials=1)

        assert isinstance(result, dict)

    def test_contains_required_keys(self, synthetic_data):
        """Test that result contains required keys."""
        returns, factors = synthetic_data

        result = compute_alpha_with_dsr(returns, factors, n_trials=1)

        required_keys = [
            "alpha_annual",
            "t_alpha",
            "sharpe_ratio",
            "dsr",
            "psr",
            "dsr_significant",
            "harvey_threshold",
        ]
        for key in required_keys:
            assert key in result

    def test_more_trials_reduces_dsr(self, synthetic_data):
        """Test that more trials reduce DSR."""
        returns, factors = synthetic_data

        result_1 = compute_alpha_with_dsr(returns, factors, n_trials=1)
        result_10 = compute_alpha_with_dsr(returns, factors, n_trials=10)

        # Same alpha, but DSR decreases with more trials
        assert abs(result_1["alpha_annual"] - result_10["alpha_annual"]) < 0.01
        assert result_10["dsr"] < result_1["dsr"]

    def test_harvey_threshold_flag(self, synthetic_data):
        """Test Harvey-Liu-Zhu threshold (t > 3.0)."""
        returns, factors = synthetic_data

        result = compute_alpha_with_dsr(returns, factors)

        # Check that harvey_threshold is bool
        assert isinstance(result["harvey_threshold"], (bool, np.bool_))

        # Should be True if |t| > 3.0
        if abs(result["t_alpha"]) > 3.0:
            assert result["harvey_threshold"]
        else:
            assert not result["harvey_threshold"]

    def test_annualize_alpha(self, synthetic_data):
        """Test alpha annualization."""
        returns, factors = synthetic_data

        # Annualized
        result_annual = compute_alpha_with_dsr(
            returns, factors, annualize=True, periods_per_year=252
        )

        # Not annualized
        result_daily = compute_alpha_with_dsr(
            returns, factors, annualize=False, periods_per_year=252
        )

        # Annualized should be ~252x daily
        ratio = result_annual["alpha_annual"] / result_daily["alpha_annual"]
        assert 200 < ratio < 300  # Approximately 252

    def test_model_selection(self, synthetic_data):
        """Test different factor models."""
        returns, factors = synthetic_data

        result_ff3 = compute_alpha_with_dsr(returns, factors, model="FF3")
        result_ff5 = compute_alpha_with_dsr(returns, factors, model="FF5")
        result_ff5_mom = compute_alpha_with_dsr(returns, factors, model="FF5_MOM")

        # All should return valid results
        assert result_ff3["model"] == "FF3"
        assert result_ff5["model"] == "FF5"
        assert result_ff5_mom["model"] == "FF5_MOM"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
