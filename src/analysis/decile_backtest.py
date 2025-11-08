"""
Decile Backtest Analysis

Rank banks by CNOI score and test if opacity predicts returns.
Methodology follows Fama-French portfolio sorts with Newey-West standard errors.

References:
- Fama & French (1992): Cross-section of expected stock returns
- Newey & West (1987): HAC covariance estimation
"""

import numpy as np
import pandas as pd

from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger
from src.utils.validation import validate_returns_schema

logger = get_logger(__name__)


def assign_deciles(
    df: pd.DataFrame, score_col: str, n_groups: int = 10, ascending: bool = True
) -> pd.DataFrame:
    """
    Assign deciles based on CNOI score.

    Args:
        df: DataFrame with scores
        score_col: Column name for scoring (e.g., 'CNOI')
        n_groups: Number of groups (10 = deciles, 4 = quartiles)
        ascending: True = D1 is lowest score, False = D1 is highest score

    Returns:
        DataFrame with added 'decile' column (1 to n_groups)

    Example:
        >>> df = assign_deciles(cnoi_df, 'CNOI', n_groups=10, ascending=True)
        >>> df.groupby('decile')['CNOI'].mean()
    """
    if score_col not in df.columns:
        raise DataValidationError(f"Missing score column '{score_col}'.")

    df = df.copy()

    # Rank by score
    df["decile"] = pd.qcut(
        df[score_col], q=n_groups, labels=range(1, n_groups + 1), duplicates="drop"
    )

    if not ascending:
        # Reverse decile numbering
        df["decile"] = n_groups + 1 - df["decile"].astype(int)

    return df


def compute_decile_returns(
    df: pd.DataFrame,
    decile_col: str = "decile",
    return_col: str = "ret_fwd",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """
    Compute equal-weighted or value-weighted decile returns.

    Args:
        df: DataFrame with deciles and returns
        decile_col: Column with decile assignments
        return_col: Column with forward returns
        weight_col: Optional column for value-weighting (e.g., 'market_cap')

    Returns:
        DataFrame with decile returns by period

    Example:
        >>> decile_ret = compute_decile_returns(df, weight_col='market_cap')
        >>> decile_ret.groupby('decile')['return'].mean()
    """
    if decile_col not in df.columns or return_col not in df.columns:
        raise DataValidationError("Dataframe missing decile or return columns.")

    if weight_col is None:
        # Equal-weighted
        decile_ret = (
            df.groupby([decile_col, "date"], observed=False)[return_col].mean().reset_index()
        )
    else:
        # Value-weighted
        df["weighted_ret"] = df[return_col] * df[weight_col]
        decile_ret = (
            df.groupby([decile_col, "date"], observed=False)
            .apply(
                lambda x: (
                    (x["weighted_ret"].sum() / x[weight_col].sum())
                    if x[weight_col].sum() > 0
                    else np.nan
                ),
                include_groups=False,
            )
            .reset_index()
        )
        decile_ret = decile_ret.rename(columns={0: return_col})

    return decile_ret


def newey_west_tstat(returns: np.ndarray, lags: int = 3) -> tuple[float, float, float]:
    """
    Compute Newey-West t-statistic for time-series returns.

    Accounts for autocorrelation in return series.

    Args:
        returns: Array of returns (T periods)
        lags: Number of lags for HAC (default: 3 for quarterly)

    Returns:
        (mean, std_nw, t_stat) tuple

    Example:
        >>> mean, se, tstat = newey_west_tstat(decile_returns, lags=3)
        >>> f"Mean: {mean:.2%}, t-stat: {tstat:.2f}"
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant

    returns = np.asarray(returns)
    returns = returns[~np.isnan(returns)]  # Drop NaNs

    if len(returns) < 2:
        return np.nan, np.nan, np.nan

    # Regress returns on constant (to get mean + NW SE)
    X = add_constant(np.ones(len(returns)))
    y = returns

    model = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    mean = model.params[0]
    se_nw = model.bse[0]
    t_stat = model.tvalues[0]

    return mean, se_nw, t_stat


def compute_long_short(
    decile_ret: pd.DataFrame,
    low_decile: int = 1,
    high_decile: int = 10,
    decile_col: str = "decile",
    return_col: str = "ret_fwd",
) -> pd.DataFrame:
    """
    Compute long-short portfolio (low CNOI - high CNOI).

    Args:
        decile_ret: DataFrame from compute_decile_returns()
        low_decile: Decile to long (default: 1 = most transparent)
        high_decile: Decile to short (default: 10 = most opaque)
        decile_col: Decile column name
        return_col: Return column name

    Returns:
        DataFrame with long-short returns by period

    Example:
        >>> ls_ret = compute_long_short(decile_ret, low_decile=1, high_decile=10)
        >>> ls_ret['return'].mean()  # Average LS spread
    """
    low_ret = decile_ret[decile_ret[decile_col] == low_decile][[return_col, "date"]]
    high_ret = decile_ret[decile_ret[decile_col] == high_decile][[return_col, "date"]]

    ls = low_ret.merge(high_ret, on="date", suffixes=("_low", "_high"))
    ls["ls_return"] = ls[f"{return_col}_low"] - ls[f"{return_col}_high"]

    return ls[["date", "ls_return"]]


def run_decile_backtest(
    cnoi_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    score_col: str = "CNOI",
    return_col: str = "ret_fwd",
    n_deciles: int = 10,
    weight_col: str | None = None,
    lags: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete decile backtest pipeline.

    Args:
        cnoi_df: DataFrame with CNOI scores
        returns_df: DataFrame with forward returns
        score_col: Column to rank on (default: 'CNOI')
        return_col: Return column (default: 'ret_fwd')
        n_deciles: Number of groups (default: 10)
        weight_col: Optional value-weighting column
        lags: Newey-West lags (default: 3 for quarterly)

    Returns:
        (decile_summary, long_short_summary) DataFrames

    Example:
        >>> summary, ls = run_decile_backtest(cnoi_df, returns_df)
        >>> summary[['decile', 'mean_ret', 't_stat']].head()
        >>> ls[['ls_mean', 'ls_tstat']]
    """
    if {"ticker", "date", score_col}.difference(cnoi_df.columns):
        raise DataValidationError("CNOI dataframe missing required columns for backtest.")

    validate_returns_schema(returns_df, required_columns=("date", "ticker", return_col))

    # Assign deciles
    df = assign_deciles(cnoi_df, score_col, n_groups=n_deciles, ascending=True)

    # Merge with returns (drop duplicate return columns if present)
    if return_col in df.columns:
        df = df.drop(columns=[return_col])
    df = df.merge(returns_df, on=["ticker", "date"], how="inner")

    # Compute decile returns
    decile_ret = compute_decile_returns(
        df, decile_col="decile", return_col=return_col, weight_col=weight_col
    )

    # Summary statistics per decile
    summary = []
    for decile in range(1, n_deciles + 1):
        dec_returns = decile_ret[decile_ret["decile"] == decile][return_col].values

        mean, se_nw, t_stat = newey_west_tstat(dec_returns, lags=lags)

        summary.append(
            {
                "decile": decile,
                "mean_ret": mean,
                "std_ret": np.nanstd(dec_returns),
                "se_nw": se_nw,
                "t_stat": t_stat,
                "sharpe": mean / np.nanstd(dec_returns) if np.nanstd(dec_returns) > 0 else np.nan,
                "n_obs": len(dec_returns),
            }
        )

    summary_df = pd.DataFrame(summary)

    # Long-short
    ls_ret = compute_long_short(decile_ret, low_decile=1, high_decile=n_deciles)
    ls_mean, ls_se, ls_t = newey_west_tstat(ls_ret["ls_return"].values, lags=lags)

    ls_summary = pd.DataFrame(
        [
            {
                "portfolio": "Long-Short (D1-D10)",
                "mean_ret": ls_mean,
                "std_ret": np.nanstd(ls_ret["ls_return"]),
                "se_nw": ls_se,
                "t_stat": ls_t,
                "sharpe": (
                    ls_mean / np.nanstd(ls_ret["ls_return"])
                    if np.nanstd(ls_ret["ls_return"]) > 0
                    else np.nan
                ),
                "n_obs": len(ls_ret),
            }
        ]
    )

    logger.info("Completed decile backtest with %s deciles.", n_deciles)
    return summary_df, ls_summary


if __name__ == "__main__":  # pragma: no cover
    # Demo with simulated data
    logger.info("=" * 60)
    logger.info("Decile Backtest Demo (Simulated Data)")
    logger.info("=" * 60)

    # Simulate CNOI scores
    np.random.seed(42)
    n_banks = 50
    n_periods = 10

    data = []
    for bank in range(n_banks):
        cnoi_base = np.random.uniform(5, 35)
        for period in range(n_periods):
            data.append(
                {
                    "ticker": f"BANK{bank:02d}",
                    "date": pd.Timestamp("2023-01-01") + pd.DateOffset(months=3 * period),
                    "CNOI": cnoi_base + np.random.normal(0, 2),
                    "ret_fwd": -0.001 * cnoi_base
                    + np.random.normal(0.02, 0.05),  # Opacity → lower returns
                    "market_cap": np.random.uniform(1e9, 100e9),
                }
            )

    df = pd.DataFrame(data)

    # Run backtest
    summary, ls = run_decile_backtest(
        df, df, score_col="CNOI", return_col="ret_fwd", weight_col="market_cap", lags=3
    )

    logger.info(
        "Decile Summary (Value-Weighted):\n%s",
        summary[["decile", "mean_ret", "t_stat", "sharpe"]].to_string(index=False),
    )

    logger.info(
        "Long-Short (D1 - D10):\n%s",
        ls[["portfolio", "mean_ret", "t_stat", "sharpe"]].to_string(index=False),
    )

    logger.info("=" * 60)
    logger.info("✓ Backtest complete!")
