"""
Tests for difference-in-differences module.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.causal_inference.difference_in_differences import (
    did_summary_table,
    dynamic_did,
    prepare_did_data,
    run_did_regression,
)


@pytest.fixture
def synthetic_panel():
    """Create synthetic panel data with known treatment effect."""
    np.random.seed(42)

    n_firms = 100
    n_quarters = 20
    quarters = pd.period_range("2018Q1", periods=n_quarters, freq="Q")

    panel_data = []
    for firm_id in range(n_firms):
        # Treatment: 50 firms treated in 2020Q1
        treated = 1 if firm_id < 50 else 0

        # Firm-specific intercept
        alpha_i = np.random.normal(0.02, 0.01)

        for quarter in quarters:
            # Post indicator
            post = 1 if quarter >= pd.Period("2020Q1", freq="Q") else 0

            # True treatment effect: -3% for treated firms after 2020Q1
            treatment_effect = -0.03 if (treated == 1 and post == 1) else 0.0

            # Generate return
            ret = alpha_i + 0.01 * post + treatment_effect + np.random.normal(0, 0.02)

            panel_data.append(
                {
                    "cik": firm_id,
                    "quarter": quarter.to_timestamp(),
                    "treated": treated,
                    "post_cecl": post,
                    "ret_next_quarter": ret,
                    "log_mcap": np.random.normal(8, 1),
                    "leverage": np.random.normal(0.5, 0.1),
                }
            )

    return pd.DataFrame(panel_data)


@pytest.fixture
def prepared_did_data(synthetic_panel):
    """Prepared DiD data."""
    return prepare_did_data(synthetic_panel)


class TestPrepareDidData:
    """Tests for prepare_did_data()."""

    def test_creates_interaction_term(self, synthetic_panel):
        """Test that treat_post interaction is created."""
        did_df = prepare_did_data(synthetic_panel)

        assert "treat_post" in did_df.columns
        assert did_df["treat_post"].dtype in [np.float64, float]

    def test_interaction_equals_product(self, synthetic_panel):
        """Test that treat_post = treated × post_cecl."""
        did_df = prepare_did_data(synthetic_panel)

        expected = did_df["treated"] * did_df["post_cecl"]
        np.testing.assert_array_almost_equal(did_df["treat_post"], expected)

    def test_count_treated_post(self, synthetic_panel):
        """Test count of treat_post observations."""
        did_df = prepare_did_data(synthetic_panel)

        # 50 treated firms × 12 post-periods (2020Q1-2022Q4)
        expected_count = 50 * 12
        actual_count = did_df["treat_post"].sum()

        assert actual_count == expected_count

    def test_raises_if_missing_treatment_col(self, synthetic_panel):
        """Test raises error if treatment column missing."""
        df = synthetic_panel.drop(columns=["treated"])

        with pytest.raises(ValueError, match="Treatment column"):
            prepare_did_data(df, treatment_col="treated")

    def test_raises_if_missing_post_col(self, synthetic_panel):
        """Test raises error if post column missing."""
        df = synthetic_panel.drop(columns=["post_cecl"])

        with pytest.raises(ValueError, match="Post column"):
            prepare_did_data(df)

    def test_raises_if_missing_outcome(self, synthetic_panel):
        """Test raises error if outcome missing."""
        df = synthetic_panel.drop(columns=["ret_next_quarter"])

        with pytest.raises(ValueError, match="Outcome column"):
            prepare_did_data(df)

    def test_custom_column_names(self, synthetic_panel):
        """Test with custom column names."""
        df = synthetic_panel.rename(
            columns={
                "treated": "cecl_adopter",
                "post_cecl": "post_2020",
                "ret_next_quarter": "quarterly_return",
            }
        )

        did_df = prepare_did_data(
            df,
            treatment_col="cecl_adopter",
            post_col="post_2020",
            outcome_col="quarterly_return",
        )

        assert "treat_post" in did_df.columns


class TestRunDidRegression:
    """Tests for run_did_regression()."""

    def test_returns_result_object(self, prepared_did_data):
        """Test that function returns a result object."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
        )

        assert hasattr(result, "params")
        assert hasattr(result, "pvalues")

    def test_did_coefficient_exists(self, prepared_did_data):
        """Test that treat_post coefficient exists."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
        )

        assert "treat_post" in result.params.index

    def test_did_coefficient_sign(self, prepared_did_data):
        """Test that DiD coefficient has correct sign (negative)."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
        )

        # True effect is -3%
        did_coef = result.params["treat_post"]
        assert did_coef < 0

    def test_did_coefficient_magnitude(self, prepared_did_data):
        """Test that DiD coefficient recovers true effect."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
        )

        did_coef = result.params["treat_post"]

        # Should be close to -0.03 (true effect)
        assert abs(did_coef - (-0.03)) < 0.01

    def test_with_controls(self, prepared_did_data):
        """Test regression with control variables."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
            controls=["log_mcap", "leverage"],
        )

        # Controls should be in params
        assert "log_mcap" in result.params.index
        assert "leverage" in result.params.index

    def test_two_way_clustering(self, prepared_did_data):
        """Test that two-way clustering works."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
            cluster_entity=True,
            cluster_time=True,
        )

        # Standard errors should be computed
        assert hasattr(result, "std_errors")
        assert result.std_errors["treat_post"] > 0

    def test_entity_fixed_effects(self, prepared_did_data):
        """Test that entity fixed effects are included."""
        result_fe = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
            entity_fe=True,
            time_fe=False,
        )

        run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
            entity_fe=False,
            time_fe=False,
        )

        # Within R-squared should be positive with entity FE
        assert result_fe.rsquared_within > 0

    def test_time_fixed_effects(self, prepared_did_data):
        """Test that time fixed effects are included."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
            time_fe=True,
        )

        # Should have time fixed effects
        # Indicated by higher R-squared
        assert result.rsquared > 0

    def test_raises_if_missing_outcome(self, prepared_did_data):
        """Test raises error if outcome column missing."""
        with pytest.raises(ValueError, match="Missing required columns"):
            run_did_regression(
                prepared_did_data,
                outcome="nonexistent_outcome",
                entity_col="cik",
                time_col="quarter",
            )

    def test_raises_if_missing_entity(self, prepared_did_data):
        """Test raises error if entity column missing."""
        with pytest.raises(ValueError, match="Missing required columns"):
            run_did_regression(
                prepared_did_data,
                outcome="ret_next_quarter",
                entity_col="nonexistent_entity",
                time_col="quarter",
            )


class TestDidSummaryTable:
    """Tests for did_summary_table()."""

    def test_returns_dataframe(self, prepared_did_data):
        """Test that summary table is a DataFrame."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
        )
        summary = did_summary_table(result)

        assert isinstance(summary, pd.DataFrame)

    def test_contains_required_columns(self, prepared_did_data):
        """Test that summary table has required columns."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
        )
        summary = did_summary_table(result)

        required_cols = ["coef", "se", "t_stat", "p_value", "95%_CI_lower", "95%_CI_upper"]
        for col in required_cols:
            assert col in summary.columns

    def test_treat_post_in_index(self, prepared_did_data):
        """Test that treat_post is in summary index."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
        )
        summary = did_summary_table(result)

        assert "treat_post" in summary.index

    def test_significance_stars(self, prepared_did_data):
        """Test that significance stars are assigned."""
        result = run_did_regression(
            prepared_did_data,
            outcome="ret_next_quarter",
            entity_col="cik",
            time_col="quarter",
        )
        summary = did_summary_table(result)

        assert "sig" in summary.columns

        # treat_post should be significant (true effect = -3%)
        sig_stars = summary.loc["treat_post", "sig"]
        assert sig_stars in ["***", "**", "*", "†"]


class TestDynamicDid:
    """Tests for dynamic_did()."""

    def test_returns_coefficients_dataframe(self, synthetic_panel):
        """Test that dynamic DiD returns coefficients DataFrame."""
        coef_df, result = dynamic_did(
            synthetic_panel,
            outcome="ret_next_quarter",
            treatment_col="treated",
            entity_col="cik",
            time_col="quarter",
            treatment_date="2020Q1",
            n_leads=4,
            n_lags=4,
        )

        assert isinstance(coef_df, pd.DataFrame)

    def test_coefficients_have_event_time(self, synthetic_panel):
        """Test that coefficients have event_time column."""
        coef_df, result = dynamic_did(
            synthetic_panel,
            outcome="ret_next_quarter",
            treatment_col="treated",
            entity_col="cik",
            time_col="quarter",
            treatment_date="2020Q1",
            n_leads=4,
            n_lags=4,
        )

        assert "event_time" in coef_df.columns

    def test_reference_period_zero_coefficient(self, synthetic_panel):
        """Test that reference period (t=-1) has zero coefficient."""
        coef_df, result = dynamic_did(
            synthetic_panel,
            outcome="ret_next_quarter",
            treatment_col="treated",
            entity_col="cik",
            time_col="quarter",
            treatment_date="2020Q1",
            n_leads=4,
            n_lags=4,
        )

        ref_coef = coef_df[coef_df["event_time"] == -1]["coef"].values[0]
        assert ref_coef == 0.0

    def test_pre_treatment_coefficients_small(self, synthetic_panel):
        """Test that pre-treatment leads are close to zero (parallel trends)."""
        coef_df, result = dynamic_did(
            synthetic_panel,
            outcome="ret_next_quarter",
            treatment_col="treated",
            entity_col="cik",
            time_col="quarter",
            treatment_date="2020Q1",
            n_leads=4,
            n_lags=4,
        )

        # Pre-treatment leads (t < 0)
        pre_coefs = coef_df[coef_df["event_time"] < 0]["coef"]

        # Should be close to zero (parallel trends)
        assert abs(pre_coefs.mean()) < 0.02

    def test_post_treatment_coefficients_negative(self, synthetic_panel):
        """Test that post-treatment lags are negative."""
        coef_df, result = dynamic_did(
            synthetic_panel,
            outcome="ret_next_quarter",
            treatment_col="treated",
            entity_col="cik",
            time_col="quarter",
            treatment_date="2020Q1",
            n_leads=4,
            n_lags=4,
        )

        # Post-treatment lags (t >= 0, excluding -1)
        post_coefs = coef_df[coef_df["event_time"] >= 0]["coef"]

        # Should be negative (true effect = -3%)
        assert post_coefs.mean() < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
