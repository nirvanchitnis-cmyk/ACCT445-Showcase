"""Fama-French factor model estimation and beta calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS

from src.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_factor_loadings(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "FF5",
    use_newey_west: bool = True,
    maxlags: int = 6,
) -> dict:
    """
    Estimate factor betas via OLS regression.

    Args:
        returns: Excess returns (r - RF) or raw returns (will be adjusted)
        factors: Factor returns (must include 'RF' column)
        model: "FF3", "FF5", or "FF5_MOM" (Carhart 4-factor)
        use_newey_west: Use HAC standard errors for autocorrelation
        maxlags: Lag length for Newey-West SEs (default: 6 for daily)

    Returns:
        {
            'alpha': float,  # Intercept (Jensen's alpha)
            'beta_mkt': float,
            'beta_smb': float,
            'beta_hml': float,
            'beta_rmw': float,  # If FF5
            'beta_cma': float,  # If FF5
            'beta_mom': float,  # If FF5_MOM
            't_alpha': float,   # t-stat for alpha
            't_beta_mkt': float, # t-stat for market beta
            'r_squared': float,
            'n_obs': int,
            'residuals': pd.Series
        }

    Example:
        >>> betas = estimate_factor_loadings(spy_returns, factors, model="FF5")
        >>> betas['beta_mkt']  # Should be ~1.0 for SPY
        >>> betas['alpha']  # Jensen's alpha
    """
    # Align returns and factors by date
    aligned = pd.DataFrame({"ret": returns}).join(factors, how="inner")

    if len(aligned) < 10:
        logger.warning("Insufficient data for factor regression (%s observations)", len(aligned))
        return _empty_factor_results(model)

    # Compute excess returns (if RF is available)
    if "RF" in aligned.columns:
        aligned["ret_excess"] = aligned["ret"] - aligned["RF"]
    else:
        logger.warning("RF (risk-free rate) not found; using raw returns")
        aligned["ret_excess"] = aligned["ret"]

    # Define factor columns based on model
    if model == "FF3":
        factor_cols = ["Mkt-RF", "SMB", "HML"]
    elif model == "FF5":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    elif model == "FF5_MOM":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
    else:
        raise ValueError(f"Invalid model: {model}. Choose from FF3, FF5, FF5_MOM")

    # Check if all factors are available
    missing = [col for col in factor_cols if col not in aligned.columns]
    if missing:
        logger.error("Missing factor columns: %s", missing)
        return _empty_factor_results(model)

    # Drop NaNs
    aligned = aligned.dropna(subset=["ret_excess"] + factor_cols)

    if len(aligned) < 10:
        logger.warning("Insufficient data after dropna (%s observations)", len(aligned))
        return _empty_factor_results(model)

    # Prepare regression
    y = aligned["ret_excess"]
    X = aligned[factor_cols]
    X = sm.add_constant(X)

    # Run OLS
    if use_newey_west:
        # Newey-West HAC standard errors
        model_fit = OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    else:
        model_fit = OLS(y, X).fit()

    # Extract results
    results = {
        "alpha": model_fit.params["const"],
        "t_alpha": model_fit.tvalues["const"],
        "p_alpha": model_fit.pvalues["const"],
        "beta_mkt": model_fit.params.get("Mkt-RF", np.nan),
        "t_beta_mkt": model_fit.tvalues.get("Mkt-RF", np.nan),
        "beta_smb": model_fit.params.get("SMB", np.nan),
        "t_beta_smb": model_fit.tvalues.get("SMB", np.nan),
        "beta_hml": model_fit.params.get("HML", np.nan),
        "t_beta_hml": model_fit.tvalues.get("HML", np.nan),
        "r_squared": model_fit.rsquared,
        "r_squared_adj": model_fit.rsquared_adj,
        "n_obs": int(model_fit.nobs),
        "residuals": pd.Series(model_fit.resid, index=aligned.index),
        "model": model,
    }

    # Add FF5-specific factors
    if model in ["FF5", "FF5_MOM"]:
        results["beta_rmw"] = model_fit.params.get("RMW", np.nan)
        results["t_beta_rmw"] = model_fit.tvalues.get("RMW", np.nan)
        results["beta_cma"] = model_fit.params.get("CMA", np.nan)
        results["t_beta_cma"] = model_fit.tvalues.get("CMA", np.nan)

    # Add Momentum factor
    if model == "FF5_MOM":
        results["beta_mom"] = model_fit.params.get("MOM", np.nan)
        results["t_beta_mom"] = model_fit.tvalues.get("MOM", np.nan)

    logger.info(
        "Estimated %s factor loadings: alpha=%.4f (t=%.2f), beta_mkt=%.4f, R²=%.3f",
        model,
        results["alpha"],
        results["t_alpha"],
        results["beta_mkt"],
        results["r_squared"],
    )

    return results


def _empty_factor_results(model: str) -> dict:
    """Return empty results dict when regression fails."""
    results = {
        "alpha": np.nan,
        "t_alpha": np.nan,
        "p_alpha": np.nan,
        "beta_mkt": np.nan,
        "t_beta_mkt": np.nan,
        "beta_smb": np.nan,
        "t_beta_smb": np.nan,
        "beta_hml": np.nan,
        "t_beta_hml": np.nan,
        "r_squared": np.nan,
        "r_squared_adj": np.nan,
        "n_obs": 0,
        "residuals": pd.Series(dtype=float),
        "model": model,
    }

    if model in ["FF5", "FF5_MOM"]:
        results["beta_rmw"] = np.nan
        results["t_beta_rmw"] = np.nan
        results["beta_cma"] = np.nan
        results["t_beta_cma"] = np.nan

    if model == "FF5_MOM":
        results["beta_mom"] = np.nan
        results["t_beta_mom"] = np.nan

    return results


def compute_expected_return(
    betas: dict, factors: pd.DataFrame, include_rf: bool = True
) -> pd.Series:
    """
    Given betas, compute expected return from factor model.

    Args:
        betas: Dictionary from estimate_factor_loadings()
        factors: Factor DataFrame with same columns used in estimation
        include_rf: Add RF (risk-free rate) to expected return

    Returns:
        Series of expected returns aligned to factor dates

    Example:
        >>> expected = compute_expected_return(betas, factors)
        >>> expected.head()
    """
    model = betas.get("model", "FF5")

    # Define factor columns based on model
    if model == "FF3":
        factor_cols = ["Mkt-RF", "SMB", "HML"]
        beta_keys = ["beta_mkt", "beta_smb", "beta_hml"]
    elif model == "FF5":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
        beta_keys = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma"]
    elif model == "FF5_MOM":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"]
        beta_keys = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "beta_mom"]
    else:
        raise ValueError(f"Invalid model: {model}")

    # Check if all factors are available
    missing = [col for col in factor_cols if col not in factors.columns]
    if missing:
        logger.error("Missing factor columns for expected return: %s", missing)
        return pd.Series(dtype=float)

    # Compute expected excess return: E[R] = beta_mkt * (Mkt-RF) + beta_smb * SMB + ...
    expected = pd.Series(0.0, index=factors.index)

    for factor_col, beta_key in zip(factor_cols, beta_keys, strict=False):
        beta = betas.get(beta_key, 0.0)
        if not np.isnan(beta):
            expected += beta * factors[factor_col]

    # Add risk-free rate if requested
    if include_rf and "RF" in factors.columns:
        expected += factors["RF"]

    return expected


def compute_abnormal_return(actual_returns: pd.Series, expected_returns: pd.Series) -> pd.Series:
    """
    Abnormal return = actual - expected (alpha).

    Args:
        actual_returns: Realized returns
        expected_returns: Expected returns from factor model

    Returns:
        Series of abnormal returns (alpha)

    Example:
        >>> abnormal = compute_abnormal_return(actual, expected)
        >>> abnormal.mean()  # Average alpha
    """
    # Align returns
    aligned = pd.DataFrame({"actual": actual_returns, "expected": expected_returns})
    aligned = aligned.dropna()

    abnormal = aligned["actual"] - aligned["expected"]

    logger.info(
        "Abnormal returns: mean=%.4f%%, std=%.4f%%",
        abnormal.mean() * 100,
        abnormal.std() * 100,
    )

    return abnormal


def rolling_beta_estimation(
    returns: pd.Series,
    factors: pd.DataFrame,
    window: int = 252,
    model: str = "FF5",
) -> pd.DataFrame:
    """
    Estimate rolling factor betas over time.

    Args:
        returns: Return series
        factors: Factor DataFrame
        window: Rolling window size (default: 252 trading days = 1 year)
        model: Factor model to use

    Returns:
        DataFrame with rolling betas (columns: alpha, beta_mkt, beta_smb, ...)

    Example:
        >>> rolling_betas = rolling_beta_estimation(spy_returns, factors, window=252)
        >>> rolling_betas[['beta_mkt', 'alpha']].plot()
    """
    aligned = pd.DataFrame({"ret": returns}).join(factors, how="inner")

    if len(aligned) < window:
        logger.warning(
            "Insufficient data for rolling estimation (need %s, have %s)", window, len(aligned)
        )
        return pd.DataFrame()

    # Define columns to track
    if model == "FF3":
        cols = ["alpha", "beta_mkt", "beta_smb", "beta_hml", "r_squared"]
    elif model == "FF5":
        cols = ["alpha", "beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "r_squared"]
    elif model == "FF5_MOM":
        cols = [
            "alpha",
            "beta_mkt",
            "beta_smb",
            "beta_hml",
            "beta_rmw",
            "beta_cma",
            "beta_mom",
            "r_squared",
        ]
    else:
        raise ValueError(f"Invalid model: {model}")

    results_list = []

    for i in range(window, len(aligned) + 1):
        window_data = aligned.iloc[i - window : i]
        window_returns = window_data["ret"]

        betas = estimate_factor_loadings(
            window_returns, factors, model=model, use_newey_west=False  # Faster for rolling
        )

        result = {col: betas.get(col, np.nan) for col in cols}
        result["date"] = aligned.index[i - 1]

        results_list.append(result)

    results_df = pd.DataFrame(results_list).set_index("date")

    logger.info("Computed rolling betas over %s windows", len(results_df))

    return results_df


__all__ = [
    "estimate_factor_loadings",
    "compute_expected_return",
    "compute_abnormal_return",
    "rolling_beta_estimation",
]
