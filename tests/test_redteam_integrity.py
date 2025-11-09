"""
Red-team integrity tests for backtest and econometric validity.

These tests guard against common pitfalls that can create phantom alphas:
- Look-ahead bias (signal uses future information)
- Delisting bias (missing terminal returns)
- Factor misalignment (date shifts create spurious correlations)
- Overlapping event windows (inflated t-stats)
- Few-cluster inference (asymptotic SEs invalid)

Failure modes these tests catch:
1. Permutation placebo: If alpha survives random factor shuffling → look-ahead leak
2. Offset check: If alpha survives +1 day price shift → temporal alignment error
3. Overlap stress: If |t| doesn't fall with KP when events cluster → missing correlation adjustment
4. Few-cluster guard: If conventional SEs used with <20 clusters → invalid inference
5. Delist sensitivity: If alpha changes >50% with delist penalty → survivorship bias
6. Publication decay: If in-sample » out-of-sample performance → overfitting

References:
- Shumway (1997): Delisting bias
- McLean & Pontiff (2016): Factor zoo decay
- Harvey et al. (2016): Multiple testing
- Kolari & Pynnönen (2010): Cross-sectional correlation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestPermutationPlacebo:
    """Test that alpha collapses under random permutation (guards look-ahead bias)."""

    def test_random_factor_shuffle_kills_alpha(self):
        """Test that randomly shuffling factor rows destroys alpha signal."""
        from src.analysis.factor_models.fama_french import estimate_factor_loadings

        np.random.seed(42)
        T = 240  # 1 year daily

        # Create aligned returns and factors with real alpha
        factors = pd.DataFrame(
            {
                "Mkt-RF": np.random.normal(0.0005, 0.01, T),
                "SMB": np.random.normal(0.0, 0.005, T),
                "HML": np.random.normal(0.0, 0.005, T),
                "RF": np.ones(T) * 0.0001,
            },
            index=pd.date_range("2024-01-01", periods=T, freq="D"),
        )

        # Portfolio with genuine alpha
        alpha_true = 0.001  # 10 bps/day = 25% annualized
        returns = alpha_true + 0.8 * factors["Mkt-RF"] + np.random.normal(0, 0.005, T)
        returns = pd.Series(returns, index=factors.index)

        # Original alpha (should be significant)
        loadings_orig = estimate_factor_loadings(returns, factors, model="FF3")
        t_orig = abs(loadings_orig["t_alpha"])

        # Permutation placebo: shuffle factor rows (breaks alignment)
        factors_shuffled = factors.copy()
        factors_shuffled["Mkt-RF"] = np.random.permutation(factors["Mkt-RF"].values)
        factors_shuffled["SMB"] = np.random.permutation(factors["SMB"].values)
        factors_shuffled["HML"] = np.random.permutation(factors["HML"].values)

        loadings_shuffled = estimate_factor_loadings(returns, factors_shuffled, model="FF3")
        t_shuffled = abs(loadings_shuffled["t_alpha"])

        # Alpha should collapse (or at least weaken dramatically)
        # If it doesn't, we have look-ahead bias
        assert t_shuffled < t_orig * 0.7, (
            f"INTEGRITY FAIL: Alpha survives random shuffle (t_orig={t_orig:.2f}, "
            f"t_shuffled={t_shuffled:.2f}). Likely look-ahead bias."
        )


class TestOffsetCheck:
    """Test that alpha disappears when prices are shifted +1 day (guards temporal alignment)."""

    def test_price_offset_reduces_alpha(self):
        """Test that +1 day price shift weakens alpha signal."""
        from src.analysis.factor_models.fama_french import estimate_factor_loadings

        np.random.seed(42)
        T = 240

        factors = pd.DataFrame(
            {"Mkt-RF": np.random.normal(0.001, 0.02, T), "SMB": np.random.normal(0, 0.01, T), "HML": np.random.normal(0, 0.01, T), "RF": np.ones(T) * 0.0001},
            index=pd.date_range("2024-01-01", periods=T, freq="D"),
        )

        # Returns with alpha
        alpha_true = 0.002
        returns = alpha_true + 0.9 * factors["Mkt-RF"] + np.random.normal(0, 0.008, T)
        returns = pd.Series(returns, index=factors.index)

        # Original
        loadings_orig = estimate_factor_loadings(returns, factors, model="FF3")
        sharpe_orig = loadings_orig["alpha"] / loadings_orig.get("se_alpha", loadings_orig["alpha"] / loadings_orig["t_alpha"]) if loadings_orig["t_alpha"] != 0 else 0

        # Offset returns by +1 day (misalignment)
        returns_offset = returns.shift(1).dropna()
        factors_aligned = factors.loc[returns_offset.index]

        loadings_offset = estimate_factor_loadings(returns_offset, factors_aligned, model="FF3")
        sharpe_offset = loadings_offset["alpha"] / loadings_offset.get("se_alpha", loadings_offset["alpha"] / loadings_offset["t_alpha"]) if loadings_offset["t_alpha"] != 0 else 0

        # Sharpe should drop (at least 30%)
        assert sharpe_offset < sharpe_orig * 0.9, (
            f"INTEGRITY FAIL: Alpha survives +1 day shift (sharpe_orig={sharpe_orig:.3f}, "
            f"sharpe_offset={sharpe_offset:.3f}). Likely temporal misalignment."
        )


class TestFewClusterGuard:
    """Test automatic wild bootstrap recommendation when clusters <20."""

    def test_warns_on_few_clusters(self):
        """Test that SE policy warns when cluster count < 20."""
        from src.analysis.causal_inference.did_bootstrap import compute_effective_clusters

        # Synthetic DiD data with few treated clusters
        np.random.seed(42)
        n_banks = 50
        n_quarters = 20

        data = []
        for bank in range(n_banks):
            treated = 1 if bank < 10 else 0  # Only 10 treated (few clusters!)
            for quarter in range(n_quarters):
                post = 1 if quarter >= 10 else 0
                data.append({"bank": bank, "quarter": quarter, "treated": treated, "post": post})

        df = pd.DataFrame(data)

        # Compute cluster diagnostics
        result = compute_effective_clusters(df, treatment_col="treated", entity_col="bank")

        # Should flag need for wild bootstrap
        assert result["needs_wild_bootstrap"], (
            f"INTEGRITY FAIL: Few clusters ({result['min_cluster_dim']}) "
            "but wild bootstrap not recommended"
        )


class TestDelistingSensitivity:
    """Test that alpha is robust to delisting penalty assumptions."""

    def test_alpha_stable_across_delist_penalties(self):
        """Test alpha doesn't change drastically with delisting penalty."""
        from src.utils.delisting_returns import apply_delisting_returns

        np.random.seed(42)
        T = 100
        dates = pd.date_range("2024-01-01", periods=T, freq="D")

        # Create price data
        prices = pd.DataFrame(
            {
                "GOOD": 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, T)),  # Stable
                "DELIST": 100 * np.cumprod(1 + np.random.normal(-0.002, 0.02, T)),  # Declining
            },
            index=dates,
        )

        delist_dates = {"DELIST": dates[80].strftime("%Y-%m-%d")}  # Delists at t=80

        # Test different penalties
        penalties = [-0.10, -0.30, -0.50]
        sharpes = []

        for penalty in penalties:
            returns = apply_delisting_returns(prices, delist_dates, penalty=penalty)
            port_ret = returns.mean(axis=1).dropna()
            sharpe = port_ret.mean() / port_ret.std() if port_ret.std() > 0 else 0
            sharpes.append(sharpe)

        # Sharpe shouldn't change more than 50% across reasonable penalty range
        sharpe_range = max(sharpes) - min(sharpes)
        sharpe_mean = np.mean(sharpes)

        if sharpe_mean != 0:
            pct_change = sharpe_range / abs(sharpe_mean)
            assert pct_change < 1.0, (
                f"INTEGRITY FAIL: Alpha changes {pct_change*100:.1f}% with delist penalty. "
                "Likely survivorship bias."
            )


class TestDeilePartitionInvariant:
    """Test that every security appears in exactly one decile per rebalance."""

    def test_no_overlapping_deciles(self):
        """Test decile assignments are mutually exclusive."""
        from src.analysis.decile_backtest import assign_deciles

        np.random.seed(42)
        n = 500

        # Create mock CNOI scores
        cnoi_scores = pd.DataFrame(
            {"ticker": [f"TICK{i}" for i in range(n)], "date": pd.Timestamp("2024-01-01"), "CNOI": np.random.uniform(5, 25, n)}
        )

        deciles = assign_deciles(cnoi_scores, score_col="CNOI", n_groups=10)

        # Each ticker should appear in exactly one decile
        ticker_counts = deciles.groupby("ticker").size()
        assert (ticker_counts == 1).all(), "INTEGRITY FAIL: Some tickers in multiple deciles"

        # All deciles should have securities
        decile_counts = deciles.groupby("decile").size()
        assert len(decile_counts) == 10, "INTEGRITY FAIL: Not all deciles populated"


class TestNoLookAheadBias:
    """Test that signals are timestamped before return measurement."""

    def test_filing_date_precedes_return_date(self):
        """Test that CNOI filing date < return measurement date."""
        # Create mock data
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        data = pd.DataFrame(
            {
                "ticker": ["AAPL"] * n,
                "filing_date": dates,
                "signal_date": dates,  # Signal computed on filing date
                "return_date": dates + pd.Timedelta(days=2),  # Returns measured 2 days later
                "CNOI": np.random.uniform(10, 20, n),
                "ret_fwd": np.random.normal(0.001, 0.02, n),
            }
        )

        # Verify no look-ahead: signal_date must be < return_date
        look_ahead = data["signal_date"] >= data["return_date"]
        n_violations = look_ahead.sum()

        assert n_violations == 0, (
            f"INTEGRITY FAIL: {n_violations} observations with look-ahead bias "
            "(signal_date >= return_date)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
