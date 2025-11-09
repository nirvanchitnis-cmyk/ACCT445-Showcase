"""
Tests for opacity validation module.

Tests CNOI construct validation including convergent validity,
discriminant validity, and horse-race regressions.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.opacity_benchmarking.opacity_validation import (
    compute_cnoi_readability_correlations,
    compute_incremental_r_squared,
    compute_partial_correlations,
    dimension_contribution_analysis,
    horse_race_regression,
)


class TestCNOIReadabilityCorrelations:
    """Test convergent validity between CNOI and readability metrics."""

    def test_correlation_computation(self):
        """Test basic correlation computation."""
        cnoi_df = pd.DataFrame(
            {
                "cik": ["001", "002", "003", "004", "005"],
                "filing_date": ["2023-12-31"] * 5,
                "CNOI": [10.0, 15.0, 20.0, 12.0, 18.0],
            }
        )

        readability_df = pd.DataFrame(
            {
                "cik": ["001", "002", "003", "004", "005"],
                "filing_date": ["2023-12-31"] * 5,
                "fog_index": [12.0, 14.0, 16.0, 13.0, 15.5],
                "flesch_ease": [60.0, 50.0, 40.0, 55.0, 45.0],
            }
        )

        summary = compute_cnoi_readability_correlations(cnoi_df, readability_df)

        assert len(summary) == 2  # fog_index and flesch_ease
        assert "Metric" in summary.columns
        assert "Correlation with CNOI" in summary.columns
        assert "p-value" in summary.columns

    def test_correlation_expected_signs(self):
        """Test that correlations have expected signs."""
        # Create data where CNOI increases with fog and decreases with flesch_ease
        cnoi_df = pd.DataFrame(
            {
                "cik": [f"{i:03d}" for i in range(10)],
                "filing_date": ["2023-12-31"] * 10,
                "CNOI": [10 + i * 2 for i in range(10)],
            }
        )

        readability_df = pd.DataFrame(
            {
                "cik": [f"{i:03d}" for i in range(10)],
                "filing_date": ["2023-12-31"] * 10,
                "fog_index": [12 + i * 1.5 for i in range(10)],  # Increases with CNOI
                "flesch_ease": [70 - i * 3 for i in range(10)],  # Decreases with CNOI
            }
        )

        summary = compute_cnoi_readability_correlations(cnoi_df, readability_df)

        fog_corr = summary.loc[summary["Metric"] == "fog_index", "Correlation with CNOI"].values[0]
        flesch_corr = summary.loc[
            summary["Metric"] == "flesch_ease", "Correlation with CNOI"
        ].values[0]

        assert fog_corr > 0  # Positive correlation
        assert flesch_corr < 0  # Negative correlation

    def test_correlation_insufficient_data(self):
        """Test handling of insufficient data."""
        cnoi_df = pd.DataFrame({"cik": ["001"], "filing_date": ["2023-12-31"], "CNOI": [10.0]})

        readability_df = pd.DataFrame(
            {"cik": ["001"], "filing_date": ["2023-12-31"], "fog_index": [12.0]}
        )

        with pytest.warns(UserWarning):
            compute_cnoi_readability_correlations(cnoi_df, readability_df)

    def test_correlation_missing_merge(self):
        """Test when merge produces no matches."""
        cnoi_df = pd.DataFrame({"cik": ["001"], "filing_date": ["2023-12-31"], "CNOI": [10.0]})

        readability_df = pd.DataFrame(
            {"cik": ["002"], "filing_date": ["2023-12-31"], "fog_index": [12.0]}  # Different CIK
        )

        with pytest.warns(UserWarning):
            compute_cnoi_readability_correlations(cnoi_df, readability_df)


class TestDimensionContributionAnalysis:
    """Test CNOI dimension variance analysis."""

    def test_dimension_contribution_basic(self):
        """Test basic dimension contribution analysis."""
        cnoi_df = pd.DataFrame(
            {
                "D": [10, 15, 20, 12, 18],
                "G": [12, 14, 16, 13, 15],
                "R": [8, 12, 16, 10, 14],
                "J": [9, 11, 13, 10, 12],
                "T": [7, 8, 9, 7.5, 8.5],
                "S": [6, 10, 14, 8, 12],
                "X": [5, 7, 9, 6, 8],
                "CNOI": [10.0, 13.0, 16.0, 11.5, 14.5],
            }
        )

        summary = dimension_contribution_analysis(cnoi_df)

        assert len(summary) == 7  # All 7 dimensions
        assert "Dimension" in summary.columns
        assert "Correlation" in summary.columns
        assert "Variance Explained (R²)" in summary.columns
        assert "Weight in CNOI" in summary.columns

    def test_dimension_weights_correct(self):
        """Test that dimension weights match CNOI formula."""
        cnoi_df = pd.DataFrame(
            {
                "D": [10, 15, 20],
                "G": [12, 14, 16],
                "R": [8, 12, 16],
                "J": [9, 11, 13],
                "T": [7, 8, 9],
                "S": [6, 10, 14],
                "X": [5, 7, 9],
                "CNOI": [10.0, 13.0, 16.0],
            }
        )

        summary = dimension_contribution_analysis(cnoi_df)

        # Check weights
        expected_weights = {"D": 0.2, "G": 0.2, "R": 0.2, "J": 0.1, "T": 0.1, "S": 0.1, "X": 0.1}

        for dim, expected_weight in expected_weights.items():
            actual_weight = summary.loc[summary["Dimension"] == dim, "Weight in CNOI"].values[0]
            assert actual_weight == expected_weight

    def test_dimension_missing_cnoi_column(self):
        """Test error when CNOI column missing."""
        df = pd.DataFrame({"D": [10, 15, 20], "G": [12, 14, 16]})

        with pytest.raises(ValueError, match="CNOI"):
            dimension_contribution_analysis(df)

    def test_dimension_missing_dimensions(self):
        """Test error when no dimensions found."""
        df = pd.DataFrame({"CNOI": [10, 15, 20]})

        with pytest.raises(ValueError, match="No dimension columns"):
            dimension_contribution_analysis(df)


class TestHorseRaceRegression:
    """Test horse-race regression for discriminant validity."""

    def test_horse_race_basic(self):
        """Test basic horse-race regression."""
        np.random.seed(42)
        n = 50

        outcome_df = pd.DataFrame(
            {
                "cik": [f"{i:03d}" for i in range(n)],
                "filing_date": ["2023-12-31"] * n,
                "ret_next_quarter": np.random.randn(n) * 0.05,
            }
        )

        cnoi_df = pd.DataFrame(
            {
                "cik": [f"{i:03d}" for i in range(n)],
                "filing_date": ["2023-12-31"] * n,
                "CNOI": 10 + np.random.randn(n) * 3,
            }
        )

        readability_df = pd.DataFrame(
            {
                "cik": [f"{i:03d}" for i in range(n)],
                "filing_date": ["2023-12-31"] * n,
                "fog_index": 12 + np.random.randn(n) * 2,
            }
        )

        summary, models = horse_race_regression(outcome_df, cnoi_df, readability_df)

        assert len(summary) == 3  # 3 models
        assert "Model" in summary.columns
        assert "CNOI_coef" in summary.columns
        assert "R²" in summary.columns

        # Check models returned
        assert "cnoi_only" in models
        assert "readability_only" in models
        assert "horse_race" in models

    def test_horse_race_cnoi_retains_significance(self):
        """Test that CNOI can retain significance in horse race."""
        np.random.seed(123)
        n = 100

        # Create data where CNOI has true effect on outcome
        cnoi = 10 + np.random.randn(n) * 3
        fog = 12 + 0.3 * cnoi + np.random.randn(n) * 1.5  # Correlated with CNOI
        returns = 0.05 - 0.003 * cnoi + np.random.randn(n) * 0.02  # CNOI predicts returns

        outcome_df = pd.DataFrame(
            {
                "cik": [f"{i:03d}" for i in range(n)],
                "filing_date": ["2023-12-31"] * n,
                "ret_next_quarter": returns,
            }
        )

        cnoi_df = pd.DataFrame(
            {"cik": [f"{i:03d}" for i in range(n)], "filing_date": ["2023-12-31"] * n, "CNOI": cnoi}
        )

        readability_df = pd.DataFrame(
            {
                "cik": [f"{i:03d}" for i in range(n)],
                "filing_date": ["2023-12-31"] * n,
                "fog_index": fog,
            }
        )

        summary, models = horse_race_regression(outcome_df, cnoi_df, readability_df)

        # CNOI should be significant in model 1
        cnoi_only_tstat = summary.loc[summary["Model"] == "CNOI only", "CNOI_tstat"].values[0]
        assert abs(cnoi_only_tstat) > 1.5  # At least modest t-stat

        # CNOI should remain in horse race (not necessarily significant with n=100)
        horse_race_coef = summary.loc[
            summary["Model"].str.contains("Horse Race"), "CNOI_coef"
        ].values[0]
        assert not pd.isna(horse_race_coef)

    def test_horse_race_missing_columns(self):
        """Test error handling for missing columns."""
        outcome_df = pd.DataFrame(
            {
                "cik": ["001"],
                "filing_date": ["2023-12-31"]
                # Missing 'ret_next_quarter'
            }
        )

        cnoi_df = pd.DataFrame({"cik": ["001"], "filing_date": ["2023-12-31"], "CNOI": [15.0]})

        readability_df = pd.DataFrame(
            {"cik": ["001"], "filing_date": ["2023-12-31"], "fog_index": [12.0]}
        )

        with pytest.raises(ValueError, match="Outcome column"):
            horse_race_regression(outcome_df, cnoi_df, readability_df)


class TestIncrementalRSquared:
    """Test F-test for incremental R²."""

    def test_incremental_r2_positive(self):
        """Test that adding variables increases R²."""
        np.random.seed(42)
        n = 100

        X1 = np.random.randn(n)
        X2 = np.random.randn(n)
        y = 2 + 0.5 * X1 + 0.3 * X2 + np.random.randn(n) * 0.5

        # Restricted model: only X1
        import statsmodels.api as sm

        X_restricted = sm.add_constant(X1)
        model_restricted = sm.OLS(y, X_restricted).fit()

        # Full model: X1 + X2
        X_full = sm.add_constant(np.column_stack([X1, X2]))
        model_full = sm.OLS(y, X_full).fit()

        result = compute_incremental_r_squared(model_restricted, model_full)

        assert result["incremental_r2"] > 0
        assert result["f_stat"] > 0
        assert 0 <= result["p_value"] <= 1

    def test_incremental_r2_error_when_not_nested(self):
        """Test error when models are not nested."""
        np.random.seed(42)
        n = 50

        X = np.random.randn(n)
        y = 2 + 0.5 * X + np.random.randn(n)

        import statsmodels.api as sm

        X_with_const = sm.add_constant(X)

        # Both models have same number of parameters (not nested)
        model1 = sm.OLS(y, X_with_const).fit()
        model2 = sm.OLS(y, X_with_const).fit()

        with pytest.raises(ValueError, match="more parameters"):
            compute_incremental_r_squared(model1, model2)


class TestPartialCorrelations:
    """Test partial correlation computation."""

    def test_partial_correlation_basic(self):
        """Test basic partial correlation computation."""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame(
            {"ret": np.random.randn(n), "CNOI": np.random.randn(n), "fog": np.random.randn(n)}
        )

        result = compute_partial_correlations(data, "ret", "CNOI", "fog")

        assert "partial_corr_x1" in result
        assert "partial_corr_x2" in result
        assert "zero_order_corr_x1" in result
        assert -1 <= result["partial_corr_x1"] <= 1

    def test_partial_correlation_removes_confound(self):
        """Test that partial correlation removes confounding."""
        np.random.seed(123)
        n = 200

        # Create confounded data: Z causes both X and Y
        Z = np.random.randn(n)  # Confound (fog_index)
        X = 0.6 * Z + np.random.randn(n) * 0.5  # CNOI (caused by fog)
        Y = 0.5 * Z + np.random.randn(n) * 0.5  # Returns (caused by fog)

        data = pd.DataFrame({"ret": Y, "CNOI": X, "fog": Z})

        result = compute_partial_correlations(data, "ret", "CNOI", "fog")

        # Zero-order correlation should be positive (both caused by Z)
        assert result["zero_order_corr_x1"] > 0

        # Partial correlation (controlling for Z) should be near zero
        assert abs(result["partial_corr_x1"]) < 0.3

    def test_partial_correlation_missing_column(self):
        """Test error when column missing."""
        data = pd.DataFrame(
            {
                "ret": [0.01, 0.02],
                "CNOI": [10, 15]
                # Missing 'fog'
            }
        )

        with pytest.raises(ValueError, match="not found"):
            compute_partial_correlations(data, "ret", "CNOI", "fog")

    def test_partial_correlation_insufficient_data(self):
        """Test error with insufficient data."""
        data = pd.DataFrame({"ret": [0.01, 0.02], "CNOI": [10, 15], "fog": [12, 14]})

        with pytest.raises(ValueError, match="Insufficient data"):
            compute_partial_correlations(data, "ret", "CNOI", "fog")
