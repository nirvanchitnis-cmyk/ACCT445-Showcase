"""
Tests for robust event study tests module.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.event_study_advanced.robust_tests import (
    bmp_standardized_test,
    corrado_rank_test,
    cumulative_abnormal_return_test,
    generalized_sign_test,
    run_all_event_tests,
    sign_test,
)


@pytest.fixture
def event_ar_matrix():
    """Create event window abnormal returns matrix."""
    np.random.seed(42)

    n_stocks = 50
    n_event_days = 10

    # Event window: day 0 has negative shock
    ar_matrix = pd.DataFrame(
        np.random.normal(0, 0.02, (n_event_days, n_stocks)),
        columns=[f"stock_{i}" for i in range(n_stocks)],
    )

    # Add event effect on day 0: -5% mean AR
    ar_matrix.iloc[0] = ar_matrix.iloc[0] - 0.05

    return ar_matrix


@pytest.fixture
def estimation_ar_matrix():
    """Create estimation window abnormal returns matrix."""
    np.random.seed(42)

    n_stocks = 50
    n_estimation_days = 120

    # Estimation window: normal returns
    ar_matrix = pd.DataFrame(
        np.random.normal(0, 0.02, (n_estimation_days, n_stocks)),
        columns=[f"stock_{i}" for i in range(n_stocks)],
    )

    return ar_matrix


class TestBmpStandardizedTest:
    """Tests for bmp_standardized_test()."""

    def test_returns_dict(self, event_ar_matrix, estimation_ar_matrix):
        """Test that function returns a dictionary."""
        result = bmp_standardized_test(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        assert isinstance(result, dict)

    def test_contains_required_keys(self, event_ar_matrix, estimation_ar_matrix):
        """Test that result contains required keys."""
        result = bmp_standardized_test(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        required_keys = ["test_stat", "p_value", "mean_sar", "mean_ar", "n_stocks", "significant"]
        for key in required_keys:
            assert key in result

    def test_detects_negative_event(self, event_ar_matrix, estimation_ar_matrix):
        """Test that BMP detects negative event day AR."""
        result = bmp_standardized_test(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        # Mean AR should be negative
        assert result["mean_ar"] < 0

    def test_significant_with_large_effect(self, event_ar_matrix, estimation_ar_matrix):
        """Test that BMP finds significance with -5% effect."""
        result = bmp_standardized_test(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        # Should be significant
        assert result["significant"]
        assert result["p_value"] < 0.05

    def test_not_significant_with_no_effect(self, estimation_ar_matrix):
        """Test that BMP finds no significance when no event effect."""
        # Use estimation window as event window (no event effect)
        result = bmp_standardized_test(estimation_ar_matrix, estimation_ar_matrix, event_idx=50)

        # Should not be significant
        assert result["p_value"] > 0.05

    def test_correct_n_stocks(self, event_ar_matrix, estimation_ar_matrix):
        """Test that n_stocks is correct."""
        result = bmp_standardized_test(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        assert result["n_stocks"] == 50

    def test_raises_if_event_idx_out_of_bounds(self, event_ar_matrix, estimation_ar_matrix):
        """Test raises error if event_idx out of bounds."""
        with pytest.raises(ValueError, match="out of bounds"):
            bmp_standardized_test(event_ar_matrix, estimation_ar_matrix, event_idx=999)

    def test_handles_missing_data(self, event_ar_matrix, estimation_ar_matrix):
        """Test handles NaN values gracefully."""
        event_ar_matrix_with_nan = event_ar_matrix.copy()
        event_ar_matrix_with_nan.iloc[0, :5] = np.nan

        result = bmp_standardized_test(event_ar_matrix_with_nan, estimation_ar_matrix, event_idx=0)

        # Should still return valid result
        assert result["n_stocks"] == 45  # 50 - 5 NaN


class TestCorradoRankTest:
    """Tests for corrado_rank_test()."""

    def test_returns_dict(self, event_ar_matrix):
        """Test that function returns a dictionary."""
        result = corrado_rank_test(event_ar_matrix, event_idx=0)

        assert isinstance(result, dict)

    def test_contains_required_keys(self, event_ar_matrix):
        """Test that result contains required keys."""
        result = corrado_rank_test(event_ar_matrix, event_idx=0)

        required_keys = [
            "z_stat",
            "p_value",
            "mean_rank",
            "expected_rank",
            "n_stocks",
            "significant",
        ]
        for key in required_keys:
            assert key in result

    def test_expected_rank_is_midpoint(self, event_ar_matrix):
        """Test that expected rank is (T+1)/2."""
        result = corrado_rank_test(event_ar_matrix, event_idx=0)

        T = len(event_ar_matrix)
        expected = (T + 1) / 2

        assert result["expected_rank"] == expected

    def test_detects_low_rank_on_negative_event(self, event_ar_matrix):
        """Test that mean rank is low on negative event day."""
        result = corrado_rank_test(event_ar_matrix, event_idx=0)

        # Mean rank should be below expected (negative event)
        assert result["mean_rank"] < result["expected_rank"]

    def test_significant_with_large_effect(self, event_ar_matrix):
        """Test that Corrado finds significance with -5% effect."""
        result = corrado_rank_test(event_ar_matrix, event_idx=0)

        # Should be significant
        assert result["significant"]

    def test_not_significant_with_no_effect(self):
        """Test that Corrado finds no significance when no event effect."""
        np.random.seed(42)
        ar_matrix = pd.DataFrame(
            np.random.normal(0, 0.02, (10, 50)),
            columns=[f"stock_{i}" for i in range(50)],
        )

        result = corrado_rank_test(ar_matrix, event_idx=5)

        # Should not be significant
        assert result["p_value"] > 0.05

    def test_raises_if_event_idx_out_of_bounds(self, event_ar_matrix):
        """Test raises error if event_idx out of bounds."""
        with pytest.raises(ValueError, match="out of bounds"):
            corrado_rank_test(event_ar_matrix, event_idx=999)


class TestSignTest:
    """Tests for sign_test()."""

    def test_returns_dict(self, event_ar_matrix):
        """Test that function returns a dictionary."""
        result = sign_test(event_ar_matrix, event_idx=0)

        assert isinstance(result, dict)

    def test_contains_required_keys(self, event_ar_matrix):
        """Test that result contains required keys."""
        result = sign_test(event_ar_matrix, event_idx=0)

        required_keys = [
            "positive_pct",
            "positive_count",
            "negative_count",
            "z_stat",
            "p_value",
            "n_stocks",
            "significant",
        ]
        for key in required_keys:
            assert key in result

    def test_positive_negative_sum_to_total(self, event_ar_matrix):
        """Test that positive + negative counts sum to total."""
        result = sign_test(event_ar_matrix, event_idx=0)

        # Note: May have zeros, so sum may be less than total
        assert result["positive_count"] + result["negative_count"] <= result["n_stocks"]

    def test_fewer_positives_on_negative_event(self, event_ar_matrix):
        """Test that fewer than 50% are positive on negative event day."""
        result = sign_test(event_ar_matrix, event_idx=0)

        # Should be less than 50% positive (negative event)
        assert result["positive_pct"] < 50.0

    def test_significant_with_large_effect(self, event_ar_matrix):
        """Test that sign test finds significance with -5% effect."""
        result = sign_test(event_ar_matrix, event_idx=0)

        # Should be significant
        assert result["significant"]

    def test_near_50_percent_with_no_effect(self):
        """Test that sign test finds ~50% positive with no event effect."""
        np.random.seed(42)
        ar_matrix = pd.DataFrame(
            np.random.normal(0, 0.02, (10, 50)),
            columns=[f"stock_{i}" for i in range(50)],
        )

        result = sign_test(ar_matrix, event_idx=5)

        # Should be near 50%
        assert 30 < result["positive_pct"] < 70

    def test_raises_if_event_idx_out_of_bounds(self, event_ar_matrix):
        """Test raises error if event_idx out of bounds."""
        with pytest.raises(ValueError, match="out of bounds"):
            sign_test(event_ar_matrix, event_idx=999)


class TestGeneralizedSignTest:
    """Tests for generalized_sign_test()."""

    def test_returns_dict(self, event_ar_matrix, estimation_ar_matrix):
        """Test that function returns a dictionary."""
        result = generalized_sign_test(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        assert isinstance(result, dict)

    def test_contains_expected_pct(self, event_ar_matrix, estimation_ar_matrix):
        """Test that result contains expected_pct."""
        result = generalized_sign_test(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        assert "expected_pct" in result

    def test_expected_pct_near_50(self, event_ar_matrix, estimation_ar_matrix):
        """Test that expected percentage is near 50% (from estimation window)."""
        result = generalized_sign_test(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        # Should be close to 50%
        assert 30 < result["expected_pct"] < 70


class TestRunAllEventTests:
    """Tests for run_all_event_tests()."""

    def test_returns_dataframe(self, event_ar_matrix, estimation_ar_matrix):
        """Test that function returns a DataFrame."""
        summary = run_all_event_tests(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        assert isinstance(summary, pd.DataFrame)

    def test_has_three_tests(self, event_ar_matrix, estimation_ar_matrix):
        """Test that summary table has 3 rows (BMP, Corrado, Sign)."""
        summary = run_all_event_tests(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        assert len(summary) == 3

    def test_contains_test_names(self, event_ar_matrix, estimation_ar_matrix):
        """Test that summary contains test names."""
        summary = run_all_event_tests(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        test_names = summary["Test"].tolist()
        assert "BMP (Cross-sectional)" in test_names
        assert "Corrado (Rank)" in test_names
        assert "Sign (Nonparametric)" in test_names

    def test_all_tests_significant_with_large_effect(self, event_ar_matrix, estimation_ar_matrix):
        """Test that all tests find significance with -5% effect."""
        summary = run_all_event_tests(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        # All should be significant
        assert (summary["Significant"] == "Yes").all()

    def test_contains_required_columns(self, event_ar_matrix, estimation_ar_matrix):
        """Test that summary has required columns."""
        summary = run_all_event_tests(event_ar_matrix, estimation_ar_matrix, event_idx=0)

        required_cols = ["Test", "Statistic", "p-value", "Significant"]
        for col in required_cols:
            assert col in summary.columns


class TestCumulativeAbnormalReturnTest:
    """Tests for cumulative_abnormal_return_test()."""

    def test_returns_dict(self, event_ar_matrix, estimation_ar_matrix):
        """Test that function returns a dictionary."""
        result = cumulative_abnormal_return_test(
            event_ar_matrix, estimation_ar_matrix, start_idx=0, end_idx=2
        )

        assert isinstance(result, dict)

    def test_contains_required_keys(self, event_ar_matrix, estimation_ar_matrix):
        """Test that result contains required keys."""
        result = cumulative_abnormal_return_test(
            event_ar_matrix, estimation_ar_matrix, start_idx=0, end_idx=2
        )

        required_keys = ["CAR", "CAR_std", "t_stat", "p_value", "n_days", "n_stocks", "significant"]
        for key in required_keys:
            assert key in result

    def test_car_is_negative_with_event(self, event_ar_matrix, estimation_ar_matrix):
        """Test that CAR is negative when event day is included."""
        result = cumulative_abnormal_return_test(
            event_ar_matrix, estimation_ar_matrix, start_idx=0, end_idx=2
        )

        # CAR should be negative (includes day 0 with -5% effect)
        assert result["CAR"] < 0

    def test_n_days_is_correct(self, event_ar_matrix, estimation_ar_matrix):
        """Test that n_days is calculated correctly."""
        result = cumulative_abnormal_return_test(
            event_ar_matrix, estimation_ar_matrix, start_idx=0, end_idx=2
        )

        # Days 0, 1, 2 = 3 days
        assert result["n_days"] == 3

    def test_significant_with_large_effect(self, event_ar_matrix, estimation_ar_matrix):
        """Test that CAR test finds significance with -5% effect."""
        result = cumulative_abnormal_return_test(
            event_ar_matrix, estimation_ar_matrix, start_idx=0, end_idx=2
        )

        # Should be significant
        assert result["significant"]

    def test_uses_all_days_if_end_none(self, event_ar_matrix, estimation_ar_matrix):
        """Test that all remaining days are used if end_idx is None."""
        result = cumulative_abnormal_return_test(
            event_ar_matrix, estimation_ar_matrix, start_idx=0, end_idx=None
        )

        # Should use all 10 days
        assert result["n_days"] == 10

    def test_raises_if_start_after_end(self, event_ar_matrix, estimation_ar_matrix):
        """Test raises error if start_idx > end_idx."""
        with pytest.raises(ValueError, match="start_idx must be"):
            cumulative_abnormal_return_test(
                event_ar_matrix, estimation_ar_matrix, start_idx=5, end_idx=2
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
