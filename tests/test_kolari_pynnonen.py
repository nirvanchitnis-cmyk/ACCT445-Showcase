"""Tests for Kolari-Pynnönen correction."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.event_study_advanced.kolari_pynnonen import (
    kp_adjusted_tstat,
    kp_caar_test,
)


@pytest.fixture
def correlated_ar_matrix():
    """AR matrix with high cross-sectional correlation (clustered event)."""
    np.random.seed(42)
    n_days = 10
    n_securities = 50

    # Common factor (e.g., market-wide shock)
    common_factor = np.random.normal(0, 0.02, n_days)

    # Individual shocks
    ar_data = {}
    for security_id in range(n_securities):
        # AR = common_factor + idiosyncratic
        idio = np.random.normal(0, 0.01, n_days)
        ar_data[f"stock_{security_id}"] = common_factor + idio

    return pd.DataFrame(ar_data)


@pytest.fixture
def uncorrelated_ar_matrix():
    """AR matrix with low cross-sectional correlation."""
    np.random.seed(42)
    n_days = 10
    n_securities = 50

    # Pure idiosyncratic (no common factor)
    ar_data = {}
    for security_id in range(n_securities):
        ar_data[f"stock_{security_id}"] = np.random.normal(0, 0.01, n_days)

    return pd.DataFrame(ar_data)


class TestKPAdjustedTstat:
    """Tests for kp_adjusted_tstat()."""

    def test_returns_dict(self, correlated_ar_matrix):
        """Test returns dictionary."""
        result = kp_adjusted_tstat(correlated_ar_matrix, event_idx=0)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, correlated_ar_matrix):
        """Test contains required keys."""
        result = kp_adjusted_tstat(correlated_ar_matrix, event_idx=0)

        required_keys = [
            "mean_ar",
            "caar",
            "t_standard",
            "t_kp",
            "p_standard",
            "p_kp",
            "rho_bar",
            "adj_factor",
            "significant_kp",
        ]
        for key in required_keys:
            assert key in result

    def test_kp_reduces_tstat_when_correlated(self, correlated_ar_matrix):
        """Test that KP reduces t-stat when ARs are correlated."""
        result = kp_adjusted_tstat(correlated_ar_matrix, event_idx=0)

        # With high correlation, KP t-stat should be lower than standard
        assert result["t_kp"] < result["t_standard"]

        # Adjustment factor should be > 1
        assert result["adj_factor"] > 1.0

        # rho_bar should be positive and substantial
        assert result["rho_bar"] > 0.1

    def test_kp_similar_when_uncorrelated(self, uncorrelated_ar_matrix):
        """Test that KP ≈ standard when ARs are uncorrelated."""
        result = kp_adjusted_tstat(uncorrelated_ar_matrix, event_idx=0)

        # With low correlation, adjustment factor should be close to 1
        # Allow wider tolerance due to sampling variation in small samples
        assert 0.5 < result["adj_factor"] < 2.0

        # rho_bar should be relatively small
        assert abs(result["rho_bar"]) < 0.5

    def test_adjustment_factor_formula(self, correlated_ar_matrix):
        """Test that adjustment factor = sqrt(1 + (N-1)*rho)."""
        result = kp_adjusted_tstat(correlated_ar_matrix, event_idx=0)

        n = result["n_securities"]
        rho = result["rho_bar"]

        expected_adj = np.sqrt(1 + (n - 1) * rho)
        assert abs(result["adj_factor"] - expected_adj) < 0.01

    def test_different_event_days(self, correlated_ar_matrix):
        """Test different event day indices."""
        for event_idx in [0, 4, 9]:
            result = kp_adjusted_tstat(correlated_ar_matrix, event_idx=event_idx)
            assert "t_kp" in result

    def test_raises_on_invalid_event_idx(self, correlated_ar_matrix):
        """Test raises error for out-of-bounds event index."""
        with pytest.raises(ValueError, match="out of bounds"):
            kp_adjusted_tstat(correlated_ar_matrix, event_idx=100)


class TestKPCAARTest:
    """Tests for kp_caar_test()."""

    def test_returns_dict(self, correlated_ar_matrix):
        """Test returns dictionary."""
        result = kp_caar_test(correlated_ar_matrix, event_window=(0, 2))
        assert isinstance(result, dict)

    def test_contains_window_length(self, correlated_ar_matrix):
        """Test contains window_length field."""
        result = kp_caar_test(correlated_ar_matrix, event_window=(0, 2))
        assert "window_length" in result
        assert result["window_length"] == 3  # Days 0, 1, 2

    def test_single_day_window(self, correlated_ar_matrix):
        """Test single-day window."""
        result_window = kp_caar_test(correlated_ar_matrix, event_window=(0, 0))

        # Should have window_length = 1
        assert result_window["window_length"] == 1

        # t_kp should be computed
        assert "t_kp" in result_window

    def test_multi_day_window(self, correlated_ar_matrix):
        """Test multi-day event window."""
        result = kp_caar_test(correlated_ar_matrix, event_window=(0, 4))

        # Window length should be 5
        assert result["window_length"] == 5

        # Should have KP adjustment
        assert "t_kp" in result
        assert "adj_factor" in result

    def test_raises_on_invalid_window(self, correlated_ar_matrix):
        """Test raises error for out-of-bounds window."""
        with pytest.raises(ValueError, match="out of bounds"):
            kp_caar_test(correlated_ar_matrix, event_window=(0, 100))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
