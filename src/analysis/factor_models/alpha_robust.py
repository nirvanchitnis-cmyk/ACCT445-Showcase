"""
Robust alpha estimation with multiple-testing corrections.

References:
- Bailey & López de Prado (2014) - Deflated Sharpe ratio
- Harvey, Liu & Zhu (2016) - Multiple testing in finance (t > 3.0 threshold)
- Fama & French (2015) - Five-factor model
- Carhart (1997) - Four-factor model
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from src.utils.logger import get_logger

logger = get_logger(__name__)


def deflated_sharpe_ratio(
    sharpe_ratio: float,
    n_observations: int,
    skewness: float = 0,
    kurtosis: float = 3,
    n_trials: int = 1,
    alpha_level: float = 0.05,
) -> dict:
    """
    Compute Deflated Sharpe Ratio (Bailey & López de Prado 2014).

    Adjusts Sharpe ratio for:
    1. Non-normality (skewness, excess kurtosis)
    2. Multiple testing (number of trials)

    The DSR accounts for the probability that an observed Sharpe ratio
    is due to chance when multiple strategies/variants were tested.

    Args:
        sharpe_ratio: Observed Sharpe ratio
        n_observations: Number of return observations
        skewness: Sample skewness (default 0 = normal)
        kurtosis: Sample kurtosis (default 3 = normal; excess kurtosis = 0)
        n_trials: Number of strategies/variants tested (for multiple testing)
        alpha_level: Significance level (default 0.05)

    Returns:
        {
            'dsr': float,           # Deflated Sharpe ratio
            'psr': float,           # Probabilistic Sharpe ratio (P[SR > 0])
            'adj_alpha': float,     # Sidak-adjusted alpha for multiple tests
            'z_stat': float,        # Adjusted z-statistic
            'threshold_sr': float,  # Min SR needed for significance
            'is_significant': bool  # True if DSR exceeds threshold
        }

    Example:
        >>> # Test alpha with 12 CNOI variants tried
        >>> dsr_result = deflated_sharpe_ratio(
        ...     sharpe_ratio=0.95,
        ...     n_observations=240,  # 5 years of weekly data
        ...     skewness=-0.2,
        ...     kurtosis=4.5,
        ...     n_trials=12
        ... )
        >>> dsr_result['psr']  # Probability of skill
    """
    if n_observations <= 2:
        raise ValueError("Need at least 3 observations for DSR calculation")

    # Step 1: Adjust for non-normality
    # Variance of SR under non-normality (Bailey & López de Prado Eq. 7)
    excess_kurtosis = kurtosis - 3
    var_sr = (1 - skewness * sharpe_ratio + ((excess_kurtosis - 1) / 4) * (sharpe_ratio**2)) / (
        n_observations - 1
    )

    if var_sr <= 0:
        logger.warning(
            "Negative variance estimate (SR=%.2f, skew=%.2f, kurt=%.2f). Using std(SR)=1/sqrt(N-1)",
            sharpe_ratio,
            skewness,
            kurtosis,
        )
        var_sr = 1 / (n_observations - 1)

    # Step 2: z-statistic for SR > 0
    z_stat = sharpe_ratio * np.sqrt(n_observations - 1) / np.sqrt(var_sr)

    # Step 3: Adjust significance level for multiple testing (Šidák correction)
    # P(at least 1 false positive in K trials) = 1 - (1 - α)^K
    # Adjusted α for single test: α_adj = 1 - (1 - α_family)^(1/K)
    if n_trials > 1:
        alpha_adj = 1 - (1 - alpha_level) ** (1 / n_trials)
    else:
        alpha_adj = alpha_level

    # Step 4: Critical z-value for adjusted alpha (two-tailed)
    z_critical = norm.ppf(1 - alpha_adj / 2)

    # Step 5: Minimum SR needed for significance
    threshold_sr = z_critical * np.sqrt(var_sr) / np.sqrt(n_observations - 1)

    # Step 6: Deflated Sharpe Ratio (relative to threshold)
    dsr = sharpe_ratio / threshold_sr if threshold_sr != 0 else np.nan

    # Step 7: Probabilistic Sharpe Ratio (P[SR > 0])
    psr = norm.cdf(z_stat)

    is_significant = dsr > 1.0 and psr > (1 - alpha_level)

    logger.info(
        "DSR: SR=%.3f → DSR=%.3f (trials=%d, PSR=%.3f, threshold=%.3f, sig=%s)",
        sharpe_ratio,
        dsr,
        n_trials,
        psr,
        threshold_sr,
        is_significant,
    )

    return {
        "dsr": dsr,
        "psr": psr,
        "adj_alpha": alpha_adj,
        "z_stat": z_stat,
        "threshold_sr": threshold_sr,
        "is_significant": is_significant,
        "n_trials": n_trials,
        "n_observations": n_observations,
    }


def rolling_alpha(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "FF5",
    window: int = 52,
    step: int = 4,
    annualize: bool = False,
    periods_per_year: int = 52,
) -> pd.DataFrame:
    """
    Compute rolling alpha over time windows.

    Guards against "it only worked in one period" criticism by showing
    alpha stability across sub-periods.

    Args:
        returns: Portfolio return series (aligned with factors)
        factors: Factor DataFrame (MKT-RF, SMB, HML, RMW, CMA, MOM, RF)
        model: "FF3", "FF5", or "Carhart"
        window: Rolling window length (default 52 weeks = 1 year)
        step: Step size between windows (default 4 weeks = monthly)
        annualize: If True, annualize alpha (default False for consistency)
        periods_per_year: Periods per year (52 for weekly, 252 for daily)

    Returns:
        DataFrame with columns:
            - date: Window end date
            - alpha: Alpha for this window (decimal)
            - t_alpha: t-statistic
            - r_squared: R²
            - n_obs: Window observations

    Example:
        >>> rolling_alphas = rolling_alpha(
        ...     decile_returns,
        ...     factors,
        ...     model="FF5",
        ...     window=52,
        ...     step=4
        ... )
        >>> rolling_alphas['alpha'].median()  # Median alpha across windows
    """
    # Select factor columns based on model
    if model == "FF3":
        factor_cols = ["Mkt-RF", "SMB", "HML"]
    elif model == "FF5":
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    elif model in ["Carhart", "FF5_MOM"]:
        # Carhart = FF3 + Momentum (alias: FF5_MOM)
        # Use "MOM" to match Ken French data library naming
        factor_cols = ["Mkt-RF", "SMB", "HML", "MOM"]
    else:
        raise ValueError(f"Unknown model: {model}. Use 'FF3', 'FF5', 'Carhart', or 'FF5_MOM'")

    # Align returns and factors
    y = returns.dropna()
    X_full = factors.loc[:, factor_cols].copy()

    # Get common index
    common_idx = y.index.intersection(X_full.index)
    if len(common_idx) < window:
        raise ValueError(f"Not enough overlapping data: {len(common_idx)} < window {window}")

    y = y.loc[common_idx]
    X_full = X_full.loc[common_idx]

    # Compute excess returns if RF available
    if "RF" in factors.columns:
        rf = factors.loc[common_idx, "RF"]
        y_excess = y - rf
    else:
        logger.warning("RF column not found. Using raw returns (not excess returns).")
        y_excess = y

    # Rolling window estimation
    results = []
    for start_idx in range(0, len(common_idx) - window + 1, step):
        end_idx = start_idx + window
        window_dates = common_idx[start_idx:end_idx]

        y_win = y_excess.loc[window_dates]
        X_win = X_full.loc[window_dates]

        # OLS regression
        X_const = add_constant(X_win)
        try:
            reg = OLS(y_win, X_const).fit()
            alpha_est = reg.params["const"]
            t_alpha = reg.tvalues["const"]
            r_sq = reg.rsquared
            n_obs = len(y_win)

            # Optionally annualize alpha
            if annualize:
                alpha_annual = alpha_est * periods_per_year
            else:
                alpha_annual = alpha_est

            results.append(
                {
                    "date": window_dates[-1],
                    "alpha": alpha_annual,
                    "t_alpha": t_alpha,
                    "r_squared": r_sq,
                    "n_obs": n_obs,
                }
            )
        except Exception as e:
            logger.warning("Rolling window regression failed at %s: %s", window_dates[-1], e)
            continue

    rolling_df = pd.DataFrame(results).set_index("date")

    logger.info(
        "Rolling alpha (%s): %d windows, median alpha=%.4f (t=%.2f)",
        model,
        len(rolling_df),
        rolling_df["alpha"].median(),
        rolling_df["t_alpha"].median(),
    )

    return rolling_df


def plot_rolling_dsr(
    rolling_alpha_df: pd.DataFrame,
    n_trials: int = 1,
    figsize: tuple = (12, 6),
    save_path: str | None = None,
) -> object:
    """
    Plot rolling Deflated Sharpe Ratio over time.

    Visualizes DSR stability across rolling windows to demonstrate that
    alpha is not a one-period fluke (Bailey & López de Prado 2014).

    Args:
        rolling_alpha_df: Output from rolling_alpha() with columns [alpha, t_alpha, r_squared]
        n_trials: Number of trials for DSR calculation (default 1)
        figsize: Figure dimensions (default 12x6)
        save_path: Optional path to save figure (PNG/SVG)

    Returns:
        matplotlib.figure.Figure

    Example:
        >>> rolling_alphas = rolling_alpha(returns, factors, window=52)
        >>> fig = plot_rolling_dsr(rolling_alphas, n_trials=12)
        >>> fig.savefig('rolling_dsr.png', dpi=300, bbox_inches='tight')
    """
    import matplotlib.pyplot as plt

    # Compute DSR for each rolling window
    dsr_values = []
    for _idx, row in rolling_alpha_df.iterrows():
        # Use t-stat to infer Sharpe ratio (approximate)
        # SR ≈ t / sqrt(n-1) for simple case
        n_obs = row.get("n_obs", 52)  # Default to 1 year
        sharpe_est = row["t_alpha"] / np.sqrt(n_obs - 1)

        # Compute DSR (simplified: assume normality for rolling windows)
        dsr_result = deflated_sharpe_ratio(
            sharpe_ratio=sharpe_est,
            n_observations=int(n_obs),
            n_trials=n_trials,
            skewness=0,
            kurtosis=3,
        )
        dsr_values.append(dsr_result["dsr"])

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(rolling_alpha_df.index, dsr_values, linewidth=2, label="Rolling DSR", color="steelblue")
    ax.axhline(
        y=1.0, color="red", linestyle="--", linewidth=1.5, label="Significance Threshold (DSR=1)"
    )
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.5)

    ax.fill_between(
        rolling_alpha_df.index,
        0,
        dsr_values,
        where=[d > 1.0 for d in dsr_values],
        alpha=0.2,
        color="green",
        label="Significant",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Deflated Sharpe Ratio", fontsize=12)
    ax.set_title(f"Rolling DSR (n_trials={n_trials})", fontsize=14, fontweight="bold")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, alpha=0.3)

    # Annotate statistics
    median_dsr = np.median(dsr_values)
    pct_significant = sum(d > 1.0 for d in dsr_values) / len(dsr_values) * 100
    textstr = f"Median DSR: {median_dsr:.2f}\n{pct_significant:.1f}% windows significant"
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved rolling DSR plot to %s", save_path)

    return fig


def compute_alpha_with_dsr(
    portfolio_returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "FF5",
    n_trials: int = 1,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute Jensen's alpha with Deflated Sharpe Ratio adjustment.

    Combines alpha estimation with multiple-testing correction to address
    "did you test 20 opacity measures and pick the best?" concern.

    Args:
        portfolio_returns: Portfolio return series
        factors: Factor DataFrame
        model: "FF3", "FF5", or "Carhart"
        n_trials: Number of CNOI variants tested (for DSR)
        annualize: Annualize alpha (default True)
        periods_per_year: Periods per year (252 for daily, 52 for weekly)

    Returns:
        {
            'alpha_annual': float,
            't_alpha': float,
            'sharpe_ratio': float,
            'dsr': float,
            'psr': float,
            'dsr_significant': bool,
            'harvey_threshold': bool,  # t > 3.0 (Harvey-Liu-Zhu)
            'n_trials': int,
            ...
        }

    Example:
        >>> alpha_dsr = compute_alpha_with_dsr(
        ...     ls_returns,
        ...     factors,
        ...     model="FF5",
        ...     n_trials=12  # Tested 12 CNOI variants
        ... )
        >>> alpha_dsr['dsr']  # Deflated SR
    """
    from src.analysis.factor_models.fama_french import estimate_factor_loadings

    # Map "Carhart" to "FF5_MOM" for compatibility with estimate_factor_loadings
    model_mapped = "FF5_MOM" if model == "Carhart" else model

    # Step 1: Estimate alpha
    loadings = estimate_factor_loadings(
        portfolio_returns,
        factors,
        model=model_mapped,
        use_newey_west=True,
        maxlags=6,
    )

    alpha_daily = loadings["alpha"]
    t_alpha = loadings["t_alpha"]
    n_obs = loadings["n_obs"]

    # Annualize alpha
    if annualize:
        alpha_annual = alpha_daily * periods_per_year
    else:
        alpha_annual = alpha_daily

    # Step 2: Compute Sharpe ratio
    # Sharpe = mean(ret) / std(ret) for the strategy
    ret_mean = portfolio_returns.mean()
    ret_std = portfolio_returns.std()
    sharpe_ratio = ret_mean / ret_std if ret_std > 0 else 0

    # Annualize Sharpe
    sharpe_annual = sharpe_ratio * np.sqrt(periods_per_year)

    # Step 3: Compute skewness and kurtosis for DSR
    skew = portfolio_returns.skew()
    kurt = portfolio_returns.kurtosis()

    # Step 4: Deflated Sharpe Ratio
    dsr_result = deflated_sharpe_ratio(
        sharpe_ratio=sharpe_annual,
        n_observations=n_obs,
        skewness=skew,
        kurtosis=kurt,
        n_trials=n_trials,
    )

    # Step 5: Harvey-Liu-Zhu threshold (t > 3.0 for robust discovery)
    harvey_threshold = abs(t_alpha) > 3.0

    return {
        "alpha_annual": alpha_annual,
        "alpha_daily": alpha_daily,
        "t_alpha": t_alpha,
        "p_alpha": loadings.get("p_alpha", np.nan),
        "sharpe_ratio": sharpe_ratio,
        "sharpe_annual": sharpe_annual,
        "skewness": skew,
        "kurtosis": kurt,
        "dsr": dsr_result["dsr"],
        "psr": dsr_result["psr"],
        "dsr_threshold_sr": dsr_result["threshold_sr"],
        "dsr_significant": dsr_result["is_significant"],
        "harvey_threshold": harvey_threshold,
        "n_trials": n_trials,
        "n_obs": n_obs,
        "r_squared": loadings["r_squared"],
        "model": model,
    }


if __name__ == "__main__":
    # Example usage
    print("Alpha robustness module loaded successfully")
    print("Available functions:")
    print("  - deflated_sharpe_ratio()")
    print("  - rolling_alpha()")
    print("  - compute_alpha_with_dsr()")
