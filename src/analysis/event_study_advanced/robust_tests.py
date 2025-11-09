"""
Robust event study test statistics.

References:
- Boehmer, Musumeci, Poulsen (1991) - Event study methodology
- Corrado (1989) - Nonparametric rank test for event studies
- Brown & Warner (1985) - Using daily stock returns
- Campbell et al. (1997) - The Econometrics of Financial Markets
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


def bmp_standardized_test(
    ar_matrix: pd.DataFrame,
    estimation_window: pd.DataFrame,
    event_idx: int = 0,
) -> dict[str, Any]:
    """
    Boehmer-Musumeci-Poulsen (BMP) cross-sectional t-test.

    Accounts for event-induced variance increase.

    Method:
        1. Standardize each stock's AR by its estimation-period std dev
        2. Compute cross-sectional mean of standardized ARs
        3. Test statistic: BMP_t = mean(SAR) / [std(SAR) / √N]

    Args:
        ar_matrix: Abnormal returns during event window (rows = days, cols = stocks)
        estimation_window: Abnormal returns during estimation period (for variance)
        event_idx: Row index of event day (0 = first day in ar_matrix)

    Returns:
        {
            'test_stat': float,  # BMP t-statistic
            'p_value': float,    # Two-tailed p-value
            'mean_sar': float,   # Mean standardized AR
            'mean_ar': float,    # Mean abnormal return (unstandardized)
            'n_stocks': int,     # Number of stocks
            'significant': bool  # True if p < 0.05
        }

    Example:
        >>> result = bmp_standardized_test(ar_event, ar_estimation, event_idx=0)
        >>> print(f"BMP t-stat: {result['test_stat']:.2f}, p={result['p_value']:.4f}")
    """
    if event_idx >= len(ar_matrix):
        raise ValueError(
            f"event_idx {event_idx} out of bounds for ar_matrix length {len(ar_matrix)}"
        )

    # Step 1: Compute std dev for each stock from estimation window
    sigma_i = estimation_window.std(axis=0, ddof=1)

    # Handle cases where sigma = 0 (no variance)
    sigma_i = sigma_i.replace(0, np.nan)

    # Step 2: Standardize event-day ARs
    ar_event = ar_matrix.iloc[event_idx]
    sar = ar_event / sigma_i

    # Drop NaN values
    sar = sar.dropna()

    if len(sar) == 0:
        logger.warning("No valid standardized ARs after removing NaN")
        return {
            "test_stat": np.nan,
            "p_value": np.nan,
            "mean_sar": np.nan,
            "mean_ar": ar_event.mean(),
            "n_stocks": 0,
            "significant": False,
        }

    # Step 3: Cross-sectional test
    mean_sar = sar.mean()
    se_sar = sar.std(ddof=1) / np.sqrt(len(sar))

    if se_sar == 0:
        logger.warning("Standard error of SAR is zero")
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat = mean_sar / se_sar
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(sar) - 1))

    significant = p_value < 0.05 if not np.isnan(p_value) else False

    logger.debug(
        "BMP test: mean_SAR=%.4f, t=%.2f, p=%.4f, N=%d",
        mean_sar,
        t_stat,
        p_value,
        len(sar),
    )

    return {
        "test_stat": t_stat,
        "p_value": p_value,
        "mean_sar": mean_sar,
        "mean_ar": ar_event.mean(),
        "n_stocks": len(sar),
        "significant": significant,
    }


def corrado_rank_test(ar_matrix: pd.DataFrame, event_idx: int = 0) -> dict[str, Any]:
    """
    Corrado (1989) nonparametric rank test.

    Robust to non-normality and event-induced variance.

    Method:
        1. Rank each stock's ARs across time (1 to T)
        2. Expected rank = (T+1)/2
        3. Test if event-day ranks deviate from expectation

    Args:
        ar_matrix: Abnormal returns matrix (rows = days, cols = stocks)
        event_idx: Row index of event day

    Returns:
        {
            'z_stat': float,     # Corrado Z-statistic
            'p_value': float,
            'mean_rank': float,  # Mean rank on event day
            'expected_rank': float,  # Expected rank under H0
            'n_stocks': int,
            'significant': bool
        }

    Example:
        >>> result = corrado_rank_test(ar_matrix, event_idx=0)
        >>> print(f"Corrado Z: {result['z_stat']:.2f}")
    """
    if event_idx >= len(ar_matrix):
        raise ValueError(f"event_idx {event_idx} out of bounds")

    # Step 1: Rank ARs for each stock (column-wise)
    # Rank method: average for ties
    ranks = ar_matrix.rank(axis=0, method="average")

    # Step 2: Event-day ranks
    r_event = ranks.iloc[event_idx]

    # Drop NaN
    r_event = r_event.dropna()

    if len(r_event) == 0:
        logger.warning("No valid ranks on event day")
        return {
            "z_stat": np.nan,
            "p_value": np.nan,
            "mean_rank": np.nan,
            "expected_rank": np.nan,
            "n_stocks": 0,
            "significant": False,
        }

    # Step 3: Compute test statistic
    T = ar_matrix.shape[0]  # Total time periods
    N = len(r_event)  # Number of stocks

    K_0 = r_event.mean()  # Mean rank on event day
    K_bar = (T + 1) / 2  # Expected rank under H0

    # Variance of rank (assuming uniform distribution)
    # Var(K_i) = (T+1)(2T+1)/6
    var_K = ((T + 1) * (2 * T + 1)) / 6

    # Standard error of mean rank
    se_K = np.sqrt(var_K / N)

    if se_K == 0:
        z_stat = np.nan
        p_value = np.nan
    else:
        z_stat = (K_0 - K_bar) / se_K
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    significant = p_value < 0.05 if not np.isnan(p_value) else False

    logger.debug(
        "Corrado test: mean_rank=%.2f, expected=%.2f, z=%.2f, p=%.4f",
        K_0,
        K_bar,
        z_stat,
        p_value,
    )

    return {
        "z_stat": z_stat,
        "p_value": p_value,
        "mean_rank": K_0,
        "expected_rank": K_bar,
        "n_stocks": N,
        "significant": significant,
    }


def sign_test(ar_matrix: pd.DataFrame, event_idx: int = 0) -> dict[str, Any]:
    """
    Nonparametric sign test.

    Tests if proportion of positive ARs on event day differs from 50%.

    Args:
        ar_matrix: Abnormal returns matrix
        event_idx: Row index of event day

    Returns:
        {
            'positive_pct': float,  # % of stocks with positive AR
            'positive_count': int,  # Number of stocks with positive AR
            'negative_count': int,  # Number of stocks with negative AR
            'z_stat': float,
            'p_value': float,
            'n_stocks': int,
            'significant': bool
        }

    Example:
        >>> result = sign_test(ar_matrix, event_idx=0)
        >>> print(f"{result['positive_pct']:.1f}% positive (expected 50%)")
    """
    if event_idx >= len(ar_matrix):
        raise ValueError(f"event_idx {event_idx} out of bounds")

    ar_event = ar_matrix.iloc[event_idx]

    # Drop NaN
    ar_event = ar_event.dropna()

    if len(ar_event) == 0:
        logger.warning("No valid ARs for sign test")
        return {
            "positive_pct": np.nan,
            "positive_count": 0,
            "negative_count": 0,
            "z_stat": np.nan,
            "p_value": np.nan,
            "n_stocks": 0,
            "significant": False,
        }

    n_positive = (ar_event > 0).sum()
    n_negative = (ar_event < 0).sum()
    N = len(ar_event)

    # Under H0: p = 0.5
    p_hat = n_positive / N

    # Z-test for proportion
    # Var(p_hat) = p(1-p)/N = 0.25/N under H0
    if N == 0:
        z_stat = np.nan
        p_value = np.nan
    else:
        z_stat = (p_hat - 0.5) / np.sqrt(0.25 / N)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    significant = p_value < 0.05 if not np.isnan(p_value) else False

    logger.debug(
        "Sign test: %d/%d positive (%.1f%%), z=%.2f, p=%.4f",
        n_positive,
        N,
        p_hat * 100,
        z_stat,
        p_value,
    )

    return {
        "positive_pct": p_hat * 100,
        "positive_count": int(n_positive),
        "negative_count": int(n_negative),
        "z_stat": z_stat,
        "p_value": p_value,
        "n_stocks": N,
        "significant": significant,
    }


def generalized_sign_test(
    ar_matrix: pd.DataFrame,
    estimation_window: pd.DataFrame,
    event_idx: int = 0,
) -> dict[str, Any]:
    """
    Generalized sign test using estimation-period baseline.

    Instead of comparing to 50%, uses fraction of positive ARs in estimation period.

    Args:
        ar_matrix: Event window ARs
        estimation_window: Estimation period ARs
        event_idx: Event day index

    Returns:
        Similar to sign_test(), but with 'expected_pct' field
    """
    if event_idx >= len(ar_matrix):
        raise ValueError(f"event_idx {event_idx} out of bounds")

    # Event day ARs
    ar_event = ar_matrix.iloc[event_idx].dropna()

    if len(ar_event) == 0:
        return {
            "positive_pct": np.nan,
            "expected_pct": np.nan,
            "z_stat": np.nan,
            "p_value": np.nan,
            "n_stocks": 0,
            "significant": False,
        }

    # Compute baseline from estimation period
    # Fraction of positive returns per stock
    baseline_positive_rate = (estimation_window > 0).mean(axis=0)

    # Expected number of positive ARs on event day
    p_0 = baseline_positive_rate.mean()

    # Observed
    n_positive = (ar_event > 0).sum()
    N = len(ar_event)
    p_hat = n_positive / N

    # Z-test
    var_p = p_0 * (1 - p_0) / N
    if var_p == 0:
        z_stat = np.nan
        p_value = np.nan
    else:
        z_stat = (p_hat - p_0) / np.sqrt(var_p)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    significant = p_value < 0.05 if not np.isnan(p_value) else False

    return {
        "positive_pct": p_hat * 100,
        "expected_pct": p_0 * 100,
        "positive_count": int(n_positive),
        "z_stat": z_stat,
        "p_value": p_value,
        "n_stocks": N,
        "significant": significant,
    }


def run_all_event_tests(
    ar_matrix: pd.DataFrame,
    estimation_window: pd.DataFrame,
    event_idx: int = 0,
) -> pd.DataFrame:
    """
    Run BMP, Corrado, and Sign tests together.

    Args:
        ar_matrix: Event window abnormal returns
        estimation_window: Estimation period abnormal returns
        event_idx: Event day index

    Returns:
        Summary table:
            | Test | Statistic | p-value | Significant |
            |------|-----------|---------|-------------|
            | BMP  | 3.42      | 0.001   | Yes         |
            | Corrado | 2.87   | 0.004   | Yes         |
            | Sign | 1.96      | 0.050   | Yes         |

    Example:
        >>> summary = run_all_event_tests(ar_event, ar_estimation)
        >>> print(summary)
    """
    logger.info("Running all robust event tests...")

    # Run tests
    bmp = bmp_standardized_test(ar_matrix, estimation_window, event_idx)
    corrado = corrado_rank_test(ar_matrix, event_idx)
    sign = sign_test(ar_matrix, event_idx)

    # Create summary table
    summary = pd.DataFrame(
        [
            {
                "Test": "BMP (Cross-sectional)",
                "Statistic": bmp["test_stat"],
                "p-value": bmp["p_value"],
                "Significant": "Yes" if bmp["significant"] else "No",
                "Mean AR": bmp["mean_ar"],
            },
            {
                "Test": "Corrado (Rank)",
                "Statistic": corrado["z_stat"],
                "p-value": corrado["p_value"],
                "Significant": "Yes" if corrado["significant"] else "No",
                "Mean AR": ar_matrix.iloc[event_idx].mean(),
            },
            {
                "Test": "Sign (Nonparametric)",
                "Statistic": sign["z_stat"],
                "p-value": sign["p_value"],
                "Significant": "Yes" if sign["significant"] else "No",
                "Mean AR": ar_matrix.iloc[event_idx].mean(),
            },
        ]
    )

    # Add diagnostic info
    summary["N"] = [bmp["n_stocks"], corrado["n_stocks"], sign["n_stocks"]]

    logger.info("Robust event tests complete:")
    logger.info("\n%s", summary.to_string(index=False))

    return summary


def cumulative_abnormal_return_test(
    ar_matrix: pd.DataFrame,
    estimation_window: pd.DataFrame,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> dict[str, Any]:
    """
    Test cumulative abnormal returns (CAR) over multi-day window.

    Uses BMP-style standardization.

    Args:
        ar_matrix: Event window ARs
        estimation_window: Estimation period ARs
        start_idx: Start of CAR window
        end_idx: End of CAR window (inclusive); if None, use all remaining days

    Returns:
        {
            'CAR': float,        # Mean CAR
            'CAR_std': float,    # Std dev of CAR
            't_stat': float,
            'p_value': float,
            'n_days': int,
            'n_stocks': int,
            'significant': bool
        }
    """
    if end_idx is None:
        end_idx = len(ar_matrix) - 1

    if start_idx > end_idx:
        raise ValueError("start_idx must be <= end_idx")

    # Sum ARs over event window (per stock)
    car_by_stock = ar_matrix.iloc[start_idx : end_idx + 1].sum(axis=0)

    # Estimation period variance
    sigma_i = estimation_window.std(axis=0, ddof=1)

    # Standardized CAR
    # Var(CAR) = L * Var(AR), where L = number of days
    L = end_idx - start_idx + 1
    scar = car_by_stock / (sigma_i * np.sqrt(L))

    # Drop NaN
    scar = scar.dropna()

    if len(scar) == 0:
        return {
            "CAR": np.nan,
            "CAR_std": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "n_days": L,
            "n_stocks": 0,
            "significant": False,
        }

    # Cross-sectional test
    mean_scar = scar.mean()
    se_scar = scar.std(ddof=1) / np.sqrt(len(scar))

    if se_scar == 0:
        t_stat = np.nan
        p_value = np.nan
    else:
        t_stat = mean_scar / se_scar
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(scar) - 1))

    # CAR statistics (unstandardized)
    mean_car = car_by_stock.mean()
    std_car = car_by_stock.std(ddof=1)

    significant = p_value < 0.05 if not np.isnan(p_value) else False

    logger.info(
        "CAR test [%d:%d]: CAR=%.4f, t=%.2f, p=%.4f",
        start_idx,
        end_idx,
        mean_car,
        t_stat,
        p_value,
    )

    return {
        "CAR": mean_car,
        "CAR_std": std_car,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_days": L,
        "n_stocks": len(scar),
        "significant": significant,
    }


if __name__ == "__main__":  # pragma: no cover
    # Demo with synthetic event data
    logger.info("Robust Event Study Tests Demo")

    np.random.seed(42)

    # Create synthetic data
    n_stocks = 50
    n_estimation_days = 120
    n_event_days = 10

    # Estimation window: normal returns
    estimation_ar = pd.DataFrame(
        np.random.normal(0, 0.02, (n_estimation_days, n_stocks)),
        columns=[f"stock_{i}" for i in range(n_stocks)],
    )

    # Event window: negative shock on day 0
    event_ar = pd.DataFrame(
        np.random.normal(0, 0.02, (n_event_days, n_stocks)),
        columns=[f"stock_{i}" for i in range(n_stocks)],
    )

    # Add event effect on day 0: -5% mean AR
    event_ar.iloc[0] = event_ar.iloc[0] - 0.05

    # Run all tests
    print("\n" + "=" * 60)
    print("Event Day 0: Robust Test Statistics")
    print("=" * 60)

    summary = run_all_event_tests(event_ar, estimation_ar, event_idx=0)
    print("\n", summary.to_string(index=False))

    # Individual test details
    print("\n" + "=" * 60)
    print("Detailed Results")
    print("=" * 60)

    bmp = bmp_standardized_test(event_ar, estimation_ar, event_idx=0)
    print("\nBMP Test:")
    print(f"  Mean SAR: {bmp['mean_sar']:.4f}")
    print(f"  t-statistic: {bmp['test_stat']:.2f}")
    print(f"  p-value: {bmp['p_value']:.4f}")

    corrado = corrado_rank_test(event_ar, event_idx=0)
    print("\nCorrado Rank Test:")
    print(f"  Mean rank: {corrado['mean_rank']:.2f}")
    print(f"  Expected rank: {corrado['expected_rank']:.2f}")
    print(f"  Z-statistic: {corrado['z_stat']:.2f}")
    print(f"  p-value: {corrado['p_value']:.4f}")

    sign = sign_test(event_ar, event_idx=0)
    print("\nSign Test:")
    print(
        f"  Positive ARs: {sign['positive_count']}/{sign['n_stocks']} ({sign['positive_pct']:.1f}%)"
    )
    print(f"  Z-statistic: {sign['z_stat']:.2f}")
    print(f"  p-value: {sign['p_value']:.4f}")

    # CAR test
    print("\n" + "=" * 60)
    print("Cumulative Abnormal Return Test (Days 0-2)")
    print("=" * 60)

    car_result = cumulative_abnormal_return_test(event_ar, estimation_ar, start_idx=0, end_idx=2)
    print(f"Mean CAR: {car_result['CAR']:.4f}")
    print(f"t-statistic: {car_result['t_stat']:.2f}")
    print(f"p-value: {car_result['p_value']:.4f}")
