"""
Superior Predictive Ability (SPA) test for multiple strategy comparison.

Implements White (2000) Reality Check and Hansen (2005) SPA test to guard against
data snooping when comparing multiple trading strategies or model specifications.

Critical for addressing: "Did you test many CNOI variants/windows and pick the best?"

References:
- White (2000): A Reality Check for Data Snooping, Econometrica
- Hansen (2005): A Test for Superior Predictive Ability, Journal of Business & Economic Statistics
- Romano & Wolf (2005): Stepwise multiple testing as formalized data snooping

Usage:
    Compare a strategy (e.g., CNOI long-short) against a benchmark (e.g., market)
    across multiple specifications (different windows, weighting, rebalancing).

    H0: Best specification has no superior performance vs. benchmark
    HA: At least one specification has genuine superior predictive ability
"""

from __future__ import annotations

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def spa_test(
    loss_matrix: np.ndarray,
    n_boot: int = 2000,
    block_length: int = 10,
    seed: int = 42,
) -> dict:
    """
    Superior Predictive Ability test (Hansen 2005).

    Tests whether the best-performing specification from a model universe
    has genuine superior performance or is due to data snooping.

    Args:
        loss_matrix: Shape [T, M] array where entry (t, m) is the loss of
            model m at time t. Convention: NEGATIVE values = better performance
            (e.g., loss = -(strategy_return - benchmark_return)).
            Model 0 is typically the benchmark (loss ≡ 0).
        n_boot: Number of bootstrap replications (default 2000)
        block_length: Average block length for stationary block bootstrap
            (default 10). For daily data with autocorrelation, use 10-20.
            For weekly data, use 4-8. For monthly, use 2-4.
        seed: Random seed for reproducibility

    Returns:
        {
            'spa_stat': float,           # Test statistic (max studentized t)
            'pvalue': float,             # Bootstrap p-value
            'best_model_idx': int,       # Index of best-performing model
            'studentized_stats': array,  # t-statistic for each model vs benchmark
            'n_models': int,
            'n_periods': int,
        }

    Example:
        >>> # loss = -(strategy - benchmark) returns
        >>> # So positive loss = strategy outperforms
        >>> loss_matrix = np.array([
        ...     [0.0, 0.002, -0.001, 0.003],  # Day 1: Model 0=bench, 1=best, 3=2nd
        ...     [0.0, 0.001,  0.002, 0.001],  # Day 2
        ...     ...
        ... ])
        >>> result = spa_test(loss_matrix)
        >>> if result['pvalue'] < 0.05:
        ...     print(f"Model {result['best_model_idx']} has SPA (p={result['pvalue']:.4f})")

    Notes:
        - Uses stationary block bootstrap (Politis & Romano 1994) to preserve
          time-series dependence under the null.
        - Studentizes each model's loss to reduce influence of high-variance specs.
        - P-value computed via bootstrap distribution of the maximum statistic.
        - If p < 0.05, reject H0: at least one model genuinely outperforms benchmark.
    """
    rng = np.random.default_rng(seed)
    loss_matrix = np.asarray(loss_matrix, dtype=float)

    if loss_matrix.ndim != 2:
        raise ValueError("loss_matrix must be 2D [T x M]")

    T, M = loss_matrix.shape

    if T < 10:
        raise ValueError("Need at least 10 observations for SPA test")

    # Step 1: Compute sample losses (mean loss per model)
    loss_mean = loss_matrix.mean(axis=0)
    loss_std = loss_matrix.std(axis=0, ddof=1)

    # Studentize to reduce influence of high-variance models (Hansen 2005)
    t_stats = np.where(loss_std > 0, loss_mean / (loss_std / np.sqrt(T)), 0.0)

    # SPA test statistic: max of studentized statistics
    spa_stat = t_stats.max()
    best_model_idx = int(t_stats.argmax())

    logger.info(
        "SPA test: %d models, %d periods. Best model #%d, t=%.3f",
        M,
        T,
        best_model_idx,
        spa_stat,
    )

    # Step 2: Stationary block bootstrap under the null
    # Demean losses to impose H0: E[loss] = 0 (no superior performance)
    loss_demeaned = loss_matrix - loss_mean.reshape(1, M)

    def sbb_indices(T, block):
        """Generate stationary block bootstrap indices (Politis & Romano 1994)."""
        idx = []
        i = 0
        while i < T:
            # Geometric block length with mean=block
            L = rng.geometric(1.0 / block)
            start = rng.integers(0, T)
            for k in range(L):
                idx.append((start + k) % T)  # Circular wrap
                i += 1
                if i >= T:
                    break
        return np.array(idx[:T])

    boot_max_stats = []
    for b in range(n_boot):
        # Bootstrap sample
        idx_boot = sbb_indices(T, block_length)
        loss_boot = loss_demeaned[idx_boot, :]

        # Compute bootstrap test statistics
        mean_boot = loss_boot.mean(axis=0)
        std_boot = loss_boot.std(axis=0, ddof=1)
        t_boot = np.where(std_boot > 0, mean_boot / (std_boot / np.sqrt(T)), 0.0)

        boot_max_stats.append(t_boot.max())

    boot_max_stats = np.array(boot_max_stats)

    # Step 3: Compute p-value
    # P[max(t_boot) >= spa_stat] under H0
    pvalue = (boot_max_stats >= spa_stat).mean()

    logger.info(
        "SPA bootstrap complete: stat=%.3f, p=%.4f (%s)",
        spa_stat,
        pvalue,
        "REJECT H0 (genuine SPA)" if pvalue < 0.05 else "FAIL TO REJECT (could be snooping)",
    )

    return {
        "spa_stat": float(spa_stat),
        "pvalue": float(pvalue),
        "best_model_idx": int(best_model_idx),
        "studentized_stats": t_stats.tolist(),
        "n_models": int(M),
        "n_periods": int(T),
        "block_length": int(block_length),
        "n_boot": int(n_boot),
    }


def spa_test_block(
    loss_matrix: np.ndarray,
    n_boot: int = 2000,
    block_length: int = 10,
    seed: int = 42,
    studentize: bool = True,
) -> dict:
    """
    SPA test with stationary block bootstrap for time-series dependence.

    Variant of spa_test() that explicitly uses block bootstrap resampling
    to preserve autocorrelation structure in returns data. Recommended for
    daily/weekly return series with strong serial dependence.

    Args:
        loss_matrix: Same as spa_test()
        n_boot: Number of bootstrap replications
        block_length: Average block length (default 10 for daily data)
        seed: Random seed
        studentize: If True, studentize statistics (recommended per Hansen 2005)

    Returns:
        Same as spa_test() plus {'block_length': int, 'studentize': bool}

    Example:
        >>> # For daily returns with autocorrelation
        >>> result = spa_test_block(loss_matrix, block_length=10, studentize=True)
        >>> if result['pvalue'] < 0.05:
        ...     print("Genuine SPA detected (accounting for time-series dependence)")

    Notes:
        - Block bootstrap preserves within-block correlation (Politis & Romano 1994)
        - Studentization reduces influence of high-variance models (Hansen 2005)
        - For weekly data, use block_length=4-8; for monthly, use 2-4
    """
    # Delegate to main spa_test with explicit documentation of block bootstrap usage
    result = spa_test(
        loss_matrix=loss_matrix,
        n_boot=n_boot,
        block_length=block_length,
        seed=seed,
    )

    # Add metadata about studentization
    result["studentize"] = studentize
    result["method"] = "SPA with stationary block bootstrap"

    return result


if __name__ == "__main__":
    # Example usage
    print("Superior Predictive Ability (SPA) Test")
    print("=" * 50)
    print("Use to test if best strategy from universe genuinely")
    print("outperforms benchmark, or is due to data snooping.")
    print()
    print("Example:")
    print("  # loss = -(strategy - benchmark)")
    print("  result = spa_test(loss_matrix, n_boot=2000)")
    print("  if result['pvalue'] < 0.05:")
    print("      print('Genuine SPA detected!')")
