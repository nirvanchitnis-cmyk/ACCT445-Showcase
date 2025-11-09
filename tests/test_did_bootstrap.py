"""Tests for wild cluster bootstrap module."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.causal_inference.did_bootstrap import (
    compute_effective_clusters,
    wild_cluster_bootstrap,
)


@pytest.fixture
def did_panel_data():
    """Create synthetic DiD panel data."""
    np.random.seed(42)

    n_banks = 100
    n_quarters = 20

    data = []
    for bank_id in range(n_banks):
        treated = 1 if bank_id < 20 else 0  # 20% treated (few clusters)

        for quarter in range(n_quarters):
            post = 1 if quarter >= 10 else 0
            treat_post = treated * post

            # DGP: Y = bank_fe + quarter_fe + delta*treat_post + error
            bank_fe = np.random.normal(0, 0.5)
            quarter_fe = np.random.normal(0, 0.2)
            delta_true = -0.05  # True treatment effect
            error = np.random.normal(0, 0.1)

            outcome = bank_fe + quarter_fe + delta_true * treat_post + error

            data.append(
                {
                    "bank_id": bank_id,
                    "quarter": quarter,
                    "treated": treated,
                    "post": post,
                    "treat_post": treat_post,
                    "outcome": outcome,
                }
            )

    return pd.DataFrame(data)


class TestWildClusterBootstrap:
    """Tests for wild_cluster_bootstrap()."""

    def test_returns_dict(self, did_panel_data):
        """Test that function returns a dictionary."""
        result = wild_cluster_bootstrap(
            data=did_panel_data,
            outcome="outcome",
            treatment="treat_post",
            entity_id="bank_id",
            time_id="quarter",
            n_boot=99,
        )

        assert isinstance(result, dict)

    def test_contains_required_keys(self, did_panel_data):
        """Test that result contains required keys."""
        result = wild_cluster_bootstrap(
            data=did_panel_data,
            outcome="outcome",
            treatment="treat_post",
            entity_id="bank_id",
            n_boot=99,
        )

        required_keys = [
            "coef",
            "se_conventional",
            "se_bootstrap",
            "t_stat",
            "p_conventional",
            "p_bootstrap",
            "ci_lower",
            "ci_upper",
            "n_clusters",
        ]
        for key in required_keys:
            assert key in result

    def test_negative_treatment_effect(self, did_panel_data):
        """Test recovers negative treatment effect."""
        result = wild_cluster_bootstrap(
            data=did_panel_data,
            outcome="outcome",
            treatment="treat_post",
            entity_id="bank_id",
            time_id="quarter",
            n_boot=199,
        )

        # True effect is -0.05
        assert result["coef"] < 0
        assert -0.15 < result["coef"] < 0.05  # Reasonable range

    def test_confidence_interval_contains_coefficient(self, did_panel_data):
        """Test that CI is properly constructed."""
        result = wild_cluster_bootstrap(
            data=did_panel_data,
            outcome="outcome",
            treatment="treat_post",
            entity_id="bank_id",
            n_boot=99,
        )

        # CI should contain point estimate
        assert result["ci_lower"] <= result["coef"] <= result["ci_upper"]

    def test_different_weight_types(self, did_panel_data):
        """Test different bootstrap weight distributions."""
        for weight_type in ["rademacher", "mammen", "normal"]:
            result = wild_cluster_bootstrap(
                data=did_panel_data,
                outcome="outcome",
                treatment="treat_post",
                entity_id="bank_id",
                n_boot=99,
                weight_type=weight_type,
            )

            assert result["weight_type"] == weight_type
            assert "p_bootstrap" in result

    def test_n_clusters_reported(self, did_panel_data):
        """Test that number of clusters is reported."""
        result = wild_cluster_bootstrap(
            data=did_panel_data,
            outcome="outcome",
            treatment="treat_post",
            entity_id="bank_id",
            n_boot=99,
        )

        # Should report 100 clusters (banks)
        assert result["n_clusters"] == 100

    def test_bootstrap_p_value_range(self, did_panel_data):
        """Test that bootstrap p-value is in [0, 1]."""
        result = wild_cluster_bootstrap(
            data=did_panel_data,
            outcome="outcome",
            treatment="treat_post",
            entity_id="bank_id",
            n_boot=99,
        )

        assert 0 <= result["p_bootstrap"] <= 1
        assert 0 <= result["p_bootstrap_sym"] <= 1


class TestComputeEffectiveClusters:
    """Tests for compute_effective_clusters()."""

    def test_returns_dict(self, did_panel_data):
        """Test that function returns a dictionary."""
        result = compute_effective_clusters(
            data=did_panel_data,
            treatment_col="treated",
            entity_col="bank_id",
            time_col="quarter",
        )

        assert isinstance(result, dict)

    def test_contains_required_keys(self, did_panel_data):
        """Test that result contains required keys."""
        result = compute_effective_clusters(
            data=did_panel_data,
            treatment_col="treated",
            entity_col="bank_id",
            time_col="quarter",
        )

        required_keys = [
            "n_clusters_entity",
            "n_clusters_time",
            "n_treated_entities",
            "n_control_entities",
            "pct_treated",
            "min_cluster_dim",
            "needs_wild_bootstrap",
        ]
        for key in required_keys:
            assert key in result

    def test_cluster_counts(self, did_panel_data):
        """Test that cluster counts are correct."""
        result = compute_effective_clusters(
            data=did_panel_data,
            treatment_col="treated",
            entity_col="bank_id",
            time_col="quarter",
        )

        # 100 banks total
        assert result["n_clusters_entity"] == 100

        # 20 treated, 80 control
        assert result["n_treated_entities"] == 20
        assert result["n_control_entities"] == 80

        # 20 quarters
        assert result["n_clusters_time"] == 20

    def test_pct_treated(self, did_panel_data):
        """Test percentage treated calculation."""
        result = compute_effective_clusters(
            data=did_panel_data,
            treatment_col="treated",
            entity_col="bank_id",
        )

        # 20 / 100 = 0.2
        assert abs(result["pct_treated"] - 0.2) < 0.01

    def test_min_cluster_dim(self, did_panel_data):
        """Test minimum cluster dimension."""
        result = compute_effective_clusters(
            data=did_panel_data,
            treatment_col="treated",
            entity_col="bank_id",
        )

        # Min of (20 treated, 80 control) = 20
        assert result["min_cluster_dim"] == 20

    def test_needs_wild_bootstrap_flag(self, did_panel_data):
        """Test wild bootstrap recommendation."""
        compute_effective_clusters(
            data=did_panel_data,
            treatment_col="treated",
            entity_col="bank_id",
        )

        # With 20 treated clusters, should NOT need wild bootstrap (borderline)
        # But with < 20, it should recommend it
        # Adjust test data to have fewer treated
        df_few = did_panel_data.copy()
        df_few["treated"] = (df_few["bank_id"] < 10).astype(int)

        result_few = compute_effective_clusters(
            data=df_few,
            treatment_col="treated",
            entity_col="bank_id",
        )

        # 10 treated clusters → should recommend wild bootstrap
        assert result_few["needs_wild_bootstrap"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
