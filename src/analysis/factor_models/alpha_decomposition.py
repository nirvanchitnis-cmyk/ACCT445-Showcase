"""Jensen's alpha and Carhart 4-factor alpha calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.factor_models.fama_french import estimate_factor_loadings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def jensen_alpha(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int = 252,
    model: str = "FF5",
) -> dict:
    """
    Compute Jensen's alpha with Newey-West SEs.

    Args:
        portfolio_returns: Portfolio return series
        factors: Factor DataFrame (must include RF)
        annualize: Convert daily alpha to annual
        periods_per_year: Number of periods per year (252 for daily)
        model: "FF3" or "FF5"

    Returns:
        {
            'alpha_annual': float,   # Annualized alpha (decimal)
            'alpha_daily': float,    # Daily alpha
            't_stat': float,         # NW t-stat
            'p_value': float,
            'model': str
        }

    Example:
        >>> alpha_results = jensen_alpha(decile_returns, factors, model="FF5")
        >>> alpha_results['alpha_annual']  # Annualized alpha
    """
    # Estimate factor loadings (includes alpha)
    loadings = estimate_factor_loadings(
        portfolio_returns, factors, model=model, use_newey_west=True, maxlags=6
    )

    alpha_daily = loadings["alpha"]
    t_stat = loadings["t_alpha"]
    p_value = loadings.get("p_alpha", np.nan)

    # Annualize alpha
    if annualize:
        # Annualized alpha = daily_alpha * periods_per_year
        alpha_annual = alpha_daily * periods_per_year
    else:
        alpha_annual = alpha_daily

    results = {
        "alpha_annual": alpha_annual,
        "alpha_daily": alpha_daily,
        "t_stat": t_stat,
        "p_value": p_value,
        "model": model,
        "n_obs": loadings["n_obs"],
        "r_squared": loadings["r_squared"],
    }

    logger.info(
        "Jensen's alpha (%s): %.2f%% annual (t=%.2f, p=%.4f)",
        model,
        alpha_annual * 100,
        t_stat,
        p_value,
    )

    return results


def carhart_alpha(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> dict:
    """
    4-factor (FF3 + Momentum) alpha.

    Args:
        portfolio_returns: Portfolio return series
        factors: Factor DataFrame (must include MOM)
        annualize: Convert daily alpha to annual
        periods_per_year: Number of periods per year (252 for daily)

    Returns:
        Same structure as jensen_alpha but with MOM factor

    Example:
        >>> alpha_results = carhart_alpha(decile_returns, factors)
        >>> alpha_results['alpha_annual']
    """
    # Use FF5_MOM model (which includes momentum)
    return jensen_alpha(
        portfolio_returns,
        factors,
        annualize=annualize,
        periods_per_year=periods_per_year,
        model="FF5_MOM",
    )


def alpha_attribution(
    portfolio_returns: pd.Series, factors: pd.DataFrame, model: str = "FF5_MOM"
) -> pd.DataFrame:
    """
    Decompose return into:
    - Alpha (unexplained)
    - Market premium (beta_mkt * Mkt-RF)
    - Size premium (beta_smb * SMB)
    - Value premium (beta_hml * HML)
    - Profitability premium (beta_rmw * RMW)
    - Investment premium (beta_cma * CMA)
    - Momentum premium (beta_mom * MOM)

    Args:
        portfolio_returns: Portfolio return series
        factors: Factor DataFrame
        model: "FF3", "FF5", or "FF5_MOM"

    Returns:
        DataFrame with columns: ['date', 'total_return', 'alpha', 'mkt_premium', ...]

    Example:
        >>> attribution = alpha_attribution(decile_returns, factors)
        >>> attribution[['total_return', 'alpha', 'mkt_premium']].head()
    """
    # Estimate factor loadings
    loadings = estimate_factor_loadings(portfolio_returns, factors, model=model)

    # Align returns and factors
    aligned = pd.DataFrame({"ret": portfolio_returns}).join(factors, how="inner")

    if len(aligned) == 0:
        logger.warning("No overlapping dates for attribution")
        return pd.DataFrame()

    # Compute excess return
    if "RF" in aligned.columns:
        aligned["ret_excess"] = aligned["ret"] - aligned["RF"]
    else:
        aligned["ret_excess"] = aligned["ret"]

    # Initialize attribution
    attribution = pd.DataFrame(index=aligned.index)
    attribution["total_return"] = aligned["ret"]
    attribution["rf"] = aligned.get("RF", 0.0)

    # Alpha contribution (constant across all periods)
    attribution["alpha"] = loadings["alpha"]

    # Factor contributions
    attribution["mkt_premium"] = loadings["beta_mkt"] * aligned["Mkt-RF"]
    attribution["smb_premium"] = loadings["beta_smb"] * aligned["SMB"]
    attribution["hml_premium"] = loadings["beta_hml"] * aligned["HML"]

    if model in ["FF5", "FF5_MOM"]:
        attribution["rmw_premium"] = loadings["beta_rmw"] * aligned["RMW"]
        attribution["cma_premium"] = loadings["beta_cma"] * aligned["CMA"]

    if model == "FF5_MOM":
        attribution["mom_premium"] = loadings["beta_mom"] * aligned["MOM"]

    # Residual (unexplained by factors)
    # residual = total_return - (rf + alpha + factor_premiums)
    factor_premium_cols = [col for col in attribution.columns if "premium" in col]
    attribution["factor_explained"] = attribution[factor_premium_cols].sum(axis=1)
    attribution["model_return"] = (
        attribution["rf"] + attribution["alpha"] + attribution["factor_explained"]
    )
    attribution["residual"] = attribution["total_return"] - attribution["model_return"]

    logger.info(
        "Attribution complete: alpha=%.4f, avg_residual=%.4f",
        attribution["alpha"].mean(),
        attribution["residual"].mean(),
    )

    return attribution


def summarize_decile_alphas(
    decile_returns: pd.DataFrame,
    factors: pd.DataFrame,
    decile_col: str = "decile",
    return_col: str = "ret_fwd",
    model: str = "FF5",
) -> pd.DataFrame:
    """
    Compute alphas for each decile portfolio.

    Args:
        decile_returns: DataFrame with decile returns (from compute_decile_returns)
        factors: Factor DataFrame
        decile_col: Column name for decile groups
        return_col: Column name for returns
        model: Factor model to use

    Returns:
        DataFrame with columns: ['decile', 'alpha_annual', 't_stat', 'p_value', ...]

    Example:
        >>> summary = summarize_decile_alphas(decile_ret, factors, model="FF5")
        >>> summary[['decile', 'alpha_annual', 't_stat']].head()
    """
    results = []

    for decile in sorted(decile_returns[decile_col].unique()):
        decile_data = decile_returns[decile_returns[decile_col] == decile]
        decile_series = decile_data.set_index("date")[return_col]

        alpha_result = jensen_alpha(decile_series, factors, model=model)

        results.append(
            {
                "decile": decile,
                "alpha_annual": alpha_result["alpha_annual"],
                "alpha_daily": alpha_result["alpha_daily"],
                "t_stat": alpha_result["t_stat"],
                "p_value": alpha_result["p_value"],
                "r_squared": alpha_result["r_squared"],
                "n_obs": alpha_result["n_obs"],
                "model": model,
            }
        )

    summary_df = pd.DataFrame(results)

    logger.info("Computed alphas for %s deciles using %s model", len(summary_df), model)

    return summary_df


def long_short_alpha(
    long_returns: pd.Series,
    short_returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "FF5",
) -> dict:
    """
    Compute alpha for long-short portfolio.

    Args:
        long_returns: Returns for long side (e.g., D1 - transparent)
        short_returns: Returns for short side (e.g., D10 - opaque)
        factors: Factor DataFrame
        model: Factor model to use

    Returns:
        Alpha results dict

    Example:
        >>> ls_alpha = long_short_alpha(d1_returns, d10_returns, factors)
        >>> ls_alpha['alpha_annual']
    """
    # Compute long-short returns
    aligned = pd.DataFrame({"long": long_returns, "short": short_returns})
    aligned = aligned.dropna()

    ls_returns = aligned["long"] - aligned["short"]

    # Compute alpha
    alpha_result = jensen_alpha(ls_returns, factors, model=model)

    logger.info(
        "Long-Short alpha (%s): %.2f%% annual (t=%.2f)",
        model,
        alpha_result["alpha_annual"] * 100,
        alpha_result["t_stat"],
    )

    return alpha_result


__all__ = [
    "jensen_alpha",
    "carhart_alpha",
    "alpha_attribution",
    "summarize_decile_alphas",
    "long_short_alpha",
]
