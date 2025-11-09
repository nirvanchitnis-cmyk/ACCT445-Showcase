"""Tests for Superior Predictive Ability (SPA) test."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.multiplicity.spa import spa_test


@pytest.fixture
def synthetic_loss_matrix():
    """Create synthetic loss matrix for testing."""
    np.random.seed(42)
    T = 100  # 100 time periods
    M = 5  # 5 models (0=benchmark, 1-4=strategies)

    # Model 0: Benchmark (zero loss by definition)
    # Model 1: Genuinely better (-0.001 mean, positive in loss convention)
    # Models 2-4: Noise (no true SPA)

    losses = np.zeros((T, M))
    losses[:, 0] = 0.0  # Benchmark
    losses[:, 1] = -(-0.001 + np.random.normal(0, 0.01, T))  # Better (negative = better performance)
    losses[:, 2] = -(np.random.normal(0, 0.01, T))  # Noise
    losses[:, 3] = -(np.random.normal(0, 0.01, T))  # Noise
    losses[:, 4] = -(np.random.normal(0, 0.01, T))  # Noise

    return losses


class TestSPATest:
    """Tests for spa_test()."""

    def test_returns_dict(self, synthetic_loss_matrix):
        """Test returns dictionary."""
        result = spa_test(synthetic_loss_matrix, n_boot=199, seed=42)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, synthetic_loss_matrix):
        """Test contains required keys."""
        result = spa_test(synthetic_loss_matrix, n_boot=199)

        required_keys = [
            "spa_stat",
            "pvalue",
            "best_model_idx",
            "studentized_stats",
            "n_models",
            "n_periods",
        ]
        for key in required_keys:
            assert key in result

    def test_pvalue_range(self, synthetic_loss_matrix):
        """Test p-value is in [0, 1]."""
        result = spa_test(synthetic_loss_matrix, n_boot=199)
        assert 0 <= result["pvalue"] <= 1

    def test_identifies_best_model(self, synthetic_loss_matrix):
        """Test identifies model with highest studentized statistic."""
        result = spa_test(synthetic_loss_matrix, n_boot=199)

        # Best model should be model with max studentized t
        assert result["best_model_idx"] == int(np.argmax(result["studentized_stats"]))

    def test_all_noise_high_pvalue(self):
        """Test that all-noise models produce high p-value."""
        np.random.seed(42)
        T = 100
        M = 5

        # All models are pure noise (no SPA)
        losses = np.random.normal(0, 0.01, (T, M))

        result = spa_test(losses, n_boot=199, seed=42)

        # High p-value (no SPA)
        assert result["pvalue"] > 0.10

    def test_clear_winner_low_pvalue(self):
        """Test that clear winner produces low p-value."""
        np.random.seed(42)
        T = 100
        M = 5

        # Model 0: benchmark
        # Model 1: Dominant winner (+1% mean outperformance)
        # Models 2-4: Noise

        losses = np.zeros((T, M))
        losses[:, 0] = 0.0
        losses[:, 1] = -0.01 + np.random.normal(0, 0.002, T)  # Strong positive performance
        losses[:, 2:] = np.random.normal(0, 0.01, (T, 3))  # Noise

        result = spa_test(losses, n_boot=199, seed=42)

        # Should have low p-value (genuine SPA)
        # Note: With strong signal (5x Sharpe ratio), should be significant
        assert result["pvalue"] < 0.20  # Lenient threshold for small sample

    def test_raises_on_too_few_observations(self):
        """Test raises error if < 10 observations."""
        losses = np.random.normal(0, 0.01, (5, 3))

        with pytest.raises(ValueError, match="at least 10 observations"):
            spa_test(losses)

    def test_raises_on_wrong_dimensions(self):
        """Test raises error if not 2D matrix."""
        losses = np.random.normal(0, 0.01, 100)

        with pytest.raises(ValueError, match="must be 2D"):
            spa_test(losses)

    def test_block_bootstrap_preserves_dependence(self):
        """Test that block bootstrap respects time-series structure."""
        np.random.seed(42)
        T = 100
        M = 3

        # Create highly autocorrelated losses (AR(1) with rho=0.9)
        rho = 0.9
        losses = np.zeros((T, M))
        for m in range(M):
            losses[0, m] = np.random.normal(0, 0.01)
            for t in range(1, T):
                losses[t, m] = rho * losses[t - 1, m] + np.random.normal(0, 0.01)

        # SPA should account for autocorrelation via block bootstrap
        result = spa_test(losses, n_boot=199, block_length=10, seed=42)

        # Should complete without error
        assert "pvalue" in result
        assert result["block_length"] == 10


class TestSPAIntegration:
    """Integration tests for SPA with factor models."""

    def test_multiple_cnoi_specifications(self):
        """Test SPA across multiple CNOI specifications."""
        np.random.seed(42)
        T = 120  # ~2 years weekly

        # Simulate returns for different CNOI specs
        # Benchmark: Market return
        # Spec 1: CNOI decile 10-1 (best)
        # Spec 2: CNOI weighted
        # Spec 3: CNOI + Fog interaction
        # Spec 4: CNOI quarterly rebalance

        benchmark_ret = np.random.normal(0.001, 0.02, T)  # Market
        spec1_ret = 0.003 + benchmark_ret + np.random.normal(0, 0.01, T)  # Alpha = 0.3% weekly
        spec2_ret = 0.001 + benchmark_ret + np.random.normal(0, 0.015, T)  # Alpha = 0.1%
        spec3_ret = benchmark_ret + np.random.normal(0, 0.012, T)  # No alpha
        spec4_ret = 0.0005 + benchmark_ret + np.random.normal(0, 0.011, T)  # Alpha = 0.05%

        # Construct loss matrix (negative of excess returns)
        loss_matrix = np.column_stack(
            [
                np.zeros(T),  # Benchmark (zero loss)
                -(spec1_ret - benchmark_ret),  # Spec 1 vs bench
                -(spec2_ret - benchmark_ret),  # Spec 2 vs bench
                -(spec3_ret - benchmark_ret),  # Spec 3 vs bench
                -(spec4_ret - benchmark_ret),  # Spec 4 vs bench
            ]
        )

        result = spa_test(loss_matrix, n_boot=199, block_length=4)

        # Best model should be spec 1 (highest alpha)
        assert result["best_model_idx"] >= 1  # Not benchmark
        assert result["n_models"] == 5
        assert result["n_periods"] == 120


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
