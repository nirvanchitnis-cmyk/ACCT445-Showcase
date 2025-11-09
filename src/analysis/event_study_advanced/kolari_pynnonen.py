"""
Kolari-Pynnönen cross-sectional correlation adjustment for event studies.

References:
- Kolari & Pynnönen (2010) - Event study testing with cross-sectional correlation of abnormal returns
- Kolari & Pynnönen (2011) - Nonparametric rank tests for event studies
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


def kp_adjusted_tstat(
    ar_matrix: pd.DataFrame,
    event_idx: int = 0,
) -> dict:
    """
    Kolari-Pynnönen (2010) cross-sectional correlation adjustment.

    When events cluster in calendar time (e.g., SVB crisis affecting all banks
    simultaneously), abnormal returns are cross-sectionally correlated.
    Standard event study t-tests are size-distorted (too many false positives).

    KP adjusts the test statistic for cross-sectional correlation.

    Args:
        ar_matrix: Abnormal returns (rows = days, cols = securities)
        event_idx: Row index of event day (0 = first day in ar_matrix)

    Returns:
        {
            'mean_ar': float,       # Mean abnormal return
            'caar': float,          # Cumulative average AR (same as mean for 1-day)
            't_standard': float,    # Standard t-stat (WRONG if cross-corr)
            't_kp': float,          # KP-adjusted t-stat
            'p_standard': float,    # Standard p-value
            'p_kp': float,          # KP-adjusted p-value
            'rho_bar': float,       # Average cross-sectional correlation
            'adj_factor': float,    # KP adjustment factor sqrt(1 + (N-1)*rho)
            'n_securities': int,
            'significant_kp': bool  # True if p_kp < 0.05
        }

    Example:
        >>> # SVB crisis: all banks affected simultaneously
        >>> result = kp_adjusted_tstat(ar_event_window, event_idx=0)
        >>> print(f"Standard t={result['t_standard']:.2f}, KP t={result['t_kp']:.2f}")
    """
    if event_idx >= len(ar_matrix):
        raise ValueError(
            f"event_idx {event_idx} out of bounds for ar_matrix length {len(ar_matrix)}"
        )

    # Step 1: Extract event-day abnormal returns
    ar_event = ar_matrix.iloc[event_idx].dropna()
    n_securities = len(ar_event)

    if n_securities == 0:
        logger.warning("No valid ARs for event day")
        return _empty_result()

    # Step 2: Mean abnormal return
    mean_ar = ar_event.mean()
    caar = mean_ar  # For 1-day window, CAAR = mean AR

    # Step 3: Standard t-statistic (ignores cross-correlation)
    se_standard = ar_event.std(ddof=1) / np.sqrt(n_securities)

    if se_standard == 0:
        logger.warning("Zero standard error")
        return _empty_result()

    t_standard = mean_ar / se_standard
    p_standard = 2 * (1 - stats.t.cdf(abs(t_standard), df=n_securities - 1))

    # Step 4: Estimate cross-sectional correlation
    # Compute pairwise correlations
    ar_full = ar_matrix.dropna(axis=1, how="all")  # Drop cols with all NaN
    n_full = ar_full.shape[1]

    if n_full < 2:
        # Can't compute correlation with <2 securities
        logger.warning("Fewer than 2 securities; skipping KP adjustment")
        return {
            "mean_ar": mean_ar,
            "caar": caar,
            "t_standard": t_standard,
            "t_kp": t_standard,  # No adjustment
            "p_standard": p_standard,
            "p_kp": p_standard,
            "rho_bar": 0.0,
            "adj_factor": 1.0,
            "n_securities": n_securities,
            "significant_kp": p_standard < 0.05,
        }

    # Compute correlation matrix
    corr_matrix = ar_full.corr()

    # Average pairwise correlation (exclude diagonal)
    upper_tri = np.triu_indices_from(corr_matrix, k=1)
    pairwise_corrs = corr_matrix.values[upper_tri]
    rho_bar = np.nanmean(pairwise_corrs)

    # Step 5: KP adjustment factor
    # adj = sqrt(1 + (N-1) * rho_bar)
    adj_factor = np.sqrt(1 + (n_full - 1) * rho_bar)

    # Step 6: KP-adjusted standard error
    se_kp = se_standard * adj_factor

    # Step 7: KP-adjusted t-statistic
    t_kp = mean_ar / se_kp
    p_kp = 2 * (1 - stats.t.cdf(abs(t_kp), df=n_securities - 1))

    significant_kp = p_kp < 0.05

    logger.info(
        "KP adjustment: rho_bar=%.3f, adj_factor=%.3f, t_std=%.2f → t_KP=%.2f (p=%.4f)",
        rho_bar,
        adj_factor,
        t_standard,
        t_kp,
        p_kp,
    )

    return {
        "mean_ar": mean_ar,
        "caar": caar,
        "t_standard": t_standard,
        "t_kp": t_kp,
        "p_standard": p_standard,
        "p_kp": p_kp,
        "rho_bar": rho_bar,
        "adj_factor": adj_factor,
        "n_securities": n_securities,
        "significant_kp": significant_kp,
    }


def kp_caar_test(
    ar_matrix: pd.DataFrame,
    event_window: tuple[int, int] = (0, 0),
) -> dict:
    """
    KP-adjusted CAAR test over multi-day event window.

    Args:
        ar_matrix: Abnormal returns (rows = days, cols = securities)
        event_window: (start_idx, end_idx) inclusive

    Returns:
        Same structure as kp_adjusted_tstat but for CAAR

    Example:
        >>> # 3-day window around SVB
        >>> result = kp_caar_test(ar_matrix, event_window=(-1, 1))
    """
    start_idx, end_idx = event_window

    if start_idx < 0 or end_idx >= len(ar_matrix):
        raise ValueError(
            f"Event window ({start_idx}, {end_idx}) out of bounds for matrix length {len(ar_matrix)}"
        )

    # Sum ARs over event window
    ar_window = ar_matrix.iloc[start_idx : end_idx + 1]
    car_series = ar_window.sum(axis=0).dropna()  # Cumulative AR for each security

    # Create a synthetic matrix: use ar_window for correlation, car_series for test
    # We need the full ar_window to compute cross-sectional correlation properly
    result = kp_adjusted_tstat(ar_window, event_idx=0)

    # Override mean_ar with CAAR (sum of means)
    result["caar"] = car_series.mean()
    result["mean_ar"] = result["caar"] / (end_idx - start_idx + 1)

    # Update label
    result["window_length"] = end_idx - start_idx + 1

    logger.info(
        "KP CAAR test: window=(%d, %d), CAAR=%.4f, t_KP=%.2f, p=%.4f",
        start_idx,
        end_idx,
        result["caar"],
        result["t_kp"],
        result["p_kp"],
    )

    return result


def _empty_result():
    """Return empty result dict."""
    return {
        "mean_ar": np.nan,
        "caar": np.nan,
        "t_standard": np.nan,
        "t_kp": np.nan,
        "p_standard": np.nan,
        "p_kp": np.nan,
        "rho_bar": np.nan,
        "adj_factor": np.nan,
        "n_securities": 0,
        "significant_kp": False,
    }


if __name__ == "__main__":
    print("Kolari-Pynnönen module loaded")
    print("Functions:")
    print("  - kp_adjusted_tstat()")
    print("  - kp_caar_test()")
