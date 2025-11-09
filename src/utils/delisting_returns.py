"""
Delisting return handling to prevent survivorship bias.

Implements conservative delisting penalties following Shumway (1997) and
Shumway & Warther (1999) to avoid upward bias in buy-and-hold alphas.

Key insight: CRSP missing delisting returns ≈ -30% on average (Shumway 1997).
Ignoring this creates spurious performance in backtest portfolios.

References:
- Shumway (1997): The Delisting Bias in CRSP Data, Journal of Finance
- Shumway & Warther (1999): The Delisting Bias in CRSP's Nasdaq Data
- Beaver, McNichols & Price (2007): Delisting Returns and Their Effect on Research
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def apply_delisting_returns(
    prices: pd.DataFrame,
    delist_dates: dict[str, str],
    penalty: float = -0.30,
) -> pd.DataFrame:
    """
    Apply delisting returns to price data to correct survivorship bias.

    Args:
        prices: DataFrame [date × ticker] with close/adjusted close prices
        delist_dates: Dict {ticker: 'YYYY-MM-DD'} of last trading date
        penalty: Terminal return on delisting (default -30% per Shumway 1997)

    Returns:
        Returns DataFrame [date × ticker] with delisting penalties applied

    Example:
        >>> delist_dates = {'SIVB': '2023-03-10', 'SBNY': '2023-03-12'}  # SVB crisis
        >>> returns = apply_delisting_returns(prices, delist_dates, penalty=-0.30)
        >>> # SIVB gets -30% return on 2023-03-10, then NaN after

    Notes:
        - Delisting date gets penalty return (e.g., -30%)
        - All dates after delisting get NaN (security no longer traded)
        - Conservative penalty prevents upward bias in long-only portfolios
        - Sensitivity analysis should test penalty ∈ [-10%, -70%]
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")

    if not delist_dates:
        logger.warning("No delisting dates provided - no adjustments made")
        return prices.pct_change()

    # Compute returns first
    returns = prices.pct_change()

    n_adjustments = 0
    for ticker, delist_date in delist_dates.items():
        if ticker not in returns.columns:
            logger.debug("Ticker %s not found in returns - skipping delisting adjustment", ticker)
            continue

        delist_date_dt = pd.to_datetime(delist_date)

        # Find delist date in index (exact or nearest)
        if delist_date_dt in returns.index:
            delist_idx = delist_date_dt
        else:
            # Find nearest date (forward fill logic)
            idx_array = returns.index
            nearest_idx = idx_array[idx_array >= delist_date_dt].min() if any(idx_array >= delist_date_dt) else None

            if nearest_idx is None:
                logger.debug(
                    "Delisting date %s for %s is after data end - no adjustment", delist_date, ticker
                )
                continue
            delist_idx = nearest_idx

        # Apply penalty return on delisting date
        returns.loc[delist_idx, ticker] = penalty

        # Zero out future returns (no trading after delisting)
        future_mask = returns.index > delist_idx
        returns.loc[future_mask, ticker] = np.nan

        n_adjustments += 1
        logger.info(
            "Applied delisting return for %s on %s: %.1f%% (then NaN)",
            ticker,
            delist_idx.strftime("%Y-%m-%d"),
            penalty * 100,
        )

    logger.info("Delisting adjustments applied: %d tickers", n_adjustments)
    return returns


def estimate_delisting_sensitivity(
    prices: pd.DataFrame,
    delist_dates: dict[str, str],
    penalties: list[float] | None = None,
) -> pd.DataFrame:
    """
    Sensitivity analysis: test multiple delisting penalty assumptions.

    Args:
        prices: Price DataFrame
        delist_dates: Delisting dates dict
        penalties: List of penalties to test (default [-10%, -30%, -50%, -70%])

    Returns:
        DataFrame with columns [penalty, mean_return, std_return, sharpe]
        showing how portfolio performance varies with delisting assumption

    Example:
        >>> sensitivity = estimate_delisting_sensitivity(prices, delist_dates)
        >>> print(sensitivity)
           penalty  mean_return  std_return  sharpe
        0    -0.10       0.0015      0.020    0.75
        1    -0.30       0.0012      0.021    0.57  # Conservative (Shumway)
        2    -0.50       0.0009      0.022    0.41
        3    -0.70       0.0006      0.023    0.26
    """
    if penalties is None:
        penalties = [-0.10, -0.30, -0.50, -0.70]

    results = []
    for pen in penalties:
        returns = apply_delisting_returns(prices, delist_dates, penalty=pen)
        port_ret = returns.mean(axis=1)  # Equal-weighted portfolio

        results.append(
            {
                "penalty": pen,
                "mean_return": float(port_ret.mean()),
                "std_return": float(port_ret.std()),
                "sharpe": float(port_ret.mean() / port_ret.std()) if port_ret.std() > 0 else np.nan,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    print("Delisting Returns Module")
    print("=" * 50)
    print("Applies Shumway (1997) delisting penalties to correct")
    print("survivorship bias in backtest alphas.")
    print()
    print("Default penalty: -30% (CRSP average for missing delisting returns)")
    print("Recommended: Sensitivity analysis with penalties ∈ [-10%, -70%]")
