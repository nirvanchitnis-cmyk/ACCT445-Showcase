"""
Standard errors policy enforcement for factor models and panel regressions.

Implements best practices from:
- Petersen (2009): Estimating standard errors in finance panel data sets
- Cameron, Gelbach & Miller (2011): Robust inference with multiway clustering
- Driscoll & Kraay (1998): Consistent covariance matrix estimation with panel data
- Newey & West (1987): Heteroskedasticity- and autocorrelation-consistent covariance matrix

Policy:
1. Time-series (portfolio alphas): Newey-West HAC with lag selection
2. Panel (firm × time): Two-way clustering (firm, time)
3. Few clusters (<20): Wild cluster bootstrap (see did_bootstrap.py)
4. Cross-sectional: Heteroskedasticity-robust (White)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.logger import get_logger

logger = get_logger(__name__)


def fit_with_policy(
    y: pd.Series | np.ndarray,
    X: pd.DataFrame | np.ndarray,
    se_type: str = "auto",
    cluster_id: pd.Series | None = None,
    time_id: pd.Series | None = None,
    maxlags: int | None = None,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit OLS with appropriate standard error correction based on data structure.

    Implements the "Standard Errors & Clustering Policy" from METHODOLOGY.md §9.4.

    Args:
        y: Dependent variable
        X: Independent variables (will add constant if not present)
        se_type: Standard error type. Options:
            - "auto": Automatic selection based on cluster_id/time_id presence
            - "hac": Newey-West HAC (for time series)
            - "twoway": Two-way clustering (firm × time)
            - "cluster": One-way clustering
            - "robust": Heteroskedasticity-robust (White)
            - "ols": Classical OLS (homoskedastic)
        cluster_id: Cluster ID for one-way or two-way clustering (e.g., firm CIK)
        time_id: Time ID for two-way clustering (e.g., quarter)
        maxlags: Number of lags for HAC (if None, auto-select based on T)

    Returns:
        statsmodels RegressionResults with appropriate covariance matrix

    Example:
        >>> # Time-series alpha regression
        >>> result = fit_with_policy(portfolio_returns, factors, se_type="hac", maxlags=6)
        >>> print(f"Alpha: {result.params[0]:.4f}, t={result.tvalues[0]:.2f} (HAC SE)")
        >>>
        >>> # Panel DiD regression
        >>> result = fit_with_policy(
        ...     y, X, se_type="twoway", cluster_id=df['cik'], time_id=df['quarter']
        ... )
        >>> print(f"Treatment effect: {result.params['treat_post']:.4f} (2-way cluster SE)")
    """
    # Add constant if not present
    if isinstance(X, pd.DataFrame):
        if "const" not in X.columns and "Intercept" not in X.columns:
            X = sm.add_constant(X, has_constant="add")
    else:
        X = sm.add_constant(X, has_constant="add")

    # Fit OLS
    model = sm.OLS(y, X)

    # Automatic SE type selection
    if se_type == "auto":
        if cluster_id is not None and time_id is not None:
            se_type = "twoway"
            logger.info("Auto-selected SE type: two-way clustering (firm × time)")
        elif cluster_id is not None:
            se_type = "cluster"
            logger.info("Auto-selected SE type: one-way clustering")
        elif len(y) > 50:  # Time series heuristic
            se_type = "hac"
            logger.info("Auto-selected SE type: HAC (time series)")
        else:
            se_type = "robust"
            logger.info("Auto-selected SE type: heteroskedasticity-robust")

    # Apply SE correction
    if se_type == "hac":
        # Newey-West HAC for time series
        if maxlags is None:
            # Rule of thumb: maxlags ≈ floor(4 * (T/100)^(2/9)) (Newey & West 1994)
            T = len(y)
            maxlags = int(np.floor(4 * (T / 100) ** (2 / 9)))
            maxlags = max(1, min(maxlags, T // 4))  # Bound to [1, T/4]

        result = model.fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        logger.info("Fitted OLS with Newey-West HAC (maxlags=%d)", maxlags)

    elif se_type == "twoway":
        # Two-way clustering (Cameron, Gelbach & Miller 2011)
        if cluster_id is None or time_id is None:
            raise ValueError("Two-way clustering requires both cluster_id and time_id")

        # Check for few clusters
        n_clusters = pd.Series(cluster_id).nunique()
        n_time = pd.Series(time_id).nunique()

        if n_clusters < 20 or n_time < 20:
            logger.warning(
                "Few clusters (firm=%d, time=%d). Use wild cluster bootstrap (did_bootstrap.py)",
                n_clusters,
                n_time,
            )

        # statsmodels two-way clustering via groups parameter
        groups = pd.DataFrame({"cluster": cluster_id, "time": time_id})
        result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
        logger.info("Fitted OLS with two-way clustering (n_firm=%d, n_time=%d)", n_clusters, n_time)

    elif se_type == "cluster":
        # One-way clustering
        if cluster_id is None:
            raise ValueError("One-way clustering requires cluster_id")

        n_clusters = pd.Series(cluster_id).nunique()
        if n_clusters < 20:
            logger.warning(
                "Few clusters (n=%d). Use wild cluster bootstrap (did_bootstrap.py)",
                n_clusters,
            )

        result = model.fit(cov_type="cluster", cov_kwds={"groups": cluster_id})
        logger.info("Fitted OLS with one-way clustering (n_clusters=%d)", n_clusters)

    elif se_type == "robust":
        # Heteroskedasticity-robust (White 1980)
        result = model.fit(cov_type="HC3")  # HC3 for finite sample performance
        logger.info("Fitted OLS with heteroskedasticity-robust SEs (White/HC3)")

    elif se_type == "ols":
        # Classical OLS (not recommended except for teaching)
        result = model.fit()
        logger.warning(
            "Fitted OLS with classical SEs (NOT ROBUST - not recommended for publication)"
        )

    else:
        raise ValueError(
            f"Unknown se_type: {se_type}. Choose from: auto, hac, twoway, cluster, robust, ols"
        )

    return result


def estimate_alpha_with_policy(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "FF5",
    se_type: str = "hac",
    maxlags: int = 6,
) -> dict:
    """
    Estimate Jensen's alpha with enforced SE policy.

    Wrapper around factor model estimation that enforces appropriate standard errors.

    Args:
        portfolio_returns: Portfolio return series
        factors: Factor DataFrame (must include Mkt-RF, SMB, HML, etc.)
        model: Factor model ("FF3", "FF5", "Carhart")
        se_type: Standard error type (default "hac" for time series)
        maxlags: HAC lags (default 6)

    Returns:
        {
            'alpha': float,
            't_alpha': float,
            'p_alpha': float,
            'se_type': str,
            'n_obs': int,
            ...
        }

    Example:
        >>> alpha_result = estimate_alpha_with_policy(
        ...     decile_returns, factors, model="FF5", se_type="hac", maxlags=6
        ... )
        >>> print(f"Alpha: {alpha_result['alpha']:.4f} (t={alpha_result['t_alpha']:.2f}, HAC)")
    """
    # Select factors based on model
    if model == "FF3":
        factor_cols = ["Mkt-RF", "SMB", "HML"]
    elif model == "FF5":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    elif model in ["Carhart", "FF5_MOM"]:
        factor_cols = ["Mkt-RF", "SMB", "HML", "MOM"]
    else:
        raise ValueError(f"Unknown model: {model}")

    # Align data
    common_idx = portfolio_returns.index.intersection(factors.index)
    y = portfolio_returns.loc[common_idx]
    X = factors.loc[common_idx, factor_cols]

    # Compute excess returns if RF available
    if "RF" in factors.columns:
        rf = factors.loc[common_idx, "RF"]
        y_excess = y - rf
    else:
        y_excess = y

    # Fit with policy
    result = fit_with_policy(y_excess, X, se_type=se_type, maxlags=maxlags)

    return {
        "alpha": float(result.params["const"]),
        "t_alpha": float(result.tvalues["const"]),
        "p_alpha": float(result.pvalues["const"]),
        "se_alpha": float(result.bse["const"]),
        "r_squared": float(result.rsquared),
        "n_obs": int(result.nobs),
        "se_type": se_type,
        "model": model,
        "factor_loadings": {col: float(result.params[col]) for col in factor_cols},
    }


if __name__ == "__main__":  # pragma: no cover - informational CLI usage
    print("Standard Errors Policy Module")
    print("==============================")
    print("SE type selection:")
    print("  - Time series (portfolio alphas): se_type='hac'")
    print("  - Panel (firm × time): se_type='twoway'")
    print("  - Few clusters (<20): Use wild_cluster_bootstrap() in did_bootstrap.py")
    print("  - Cross-sectional: se_type='robust'")
