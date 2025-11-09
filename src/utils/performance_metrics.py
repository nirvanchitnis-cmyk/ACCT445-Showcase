"""Portfolio performance and risk metrics utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _validate_returns(returns: pd.Series, min_length: int = 1) -> pd.Series:
    series = returns.dropna()
    if len(series) < min_length:
        raise ValueError(f"Need at least {min_length} return observations.")
    return series


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Compound the period returns into an annualized figure."""
    series = _validate_returns(returns)
    growth = (1 + series).prod()

    if growth <= 0:
        logger.warning("Non-positive cumulative return; using arithmetic approximation.")
        return series.mean() * periods_per_year

    years = len(series) / periods_per_year
    return growth ** (1 / years) - 1


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns."""
    series = _validate_returns(returns, min_length=2)
    return series.std(ddof=1) * math.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Excess return over standard deviation."""
    series = _validate_returns(returns, min_length=2)
    per_period_rf = risk_free_rate / periods_per_year
    excess = series - per_period_rf
    std = excess.std(ddof=1)
    if std == 0:
        return np.inf if excess.mean() > 0 else -np.inf
    return excess.mean() * periods_per_year / std


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Excess return divided by downside deviation."""
    series = _validate_returns(returns, min_length=2)
    per_period_rf = risk_free_rate / periods_per_year
    excess = series - per_period_rf
    downside = excess[excess < 0]
    if downside.empty:
        return np.inf
    downside_dev = downside.std(ddof=1) * math.sqrt(periods_per_year)
    if downside_dev == 0:
        return np.inf
    return excess.mean() * periods_per_year / downside_dev


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline."""
    series = _validate_returns(returns)
    cumulative = (1 + series).cumprod()
    running_max = cumulative.cummax()
    drawdowns = cumulative / running_max - 1
    return drawdowns.min()


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized return divided by absolute max drawdown."""
    ann_ret = annualized_return(returns, periods_per_year)
    drawdown = abs(max_drawdown(returns))
    if drawdown == 0:
        return np.inf
    return ann_ret / drawdown


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """Active return divided by tracking error."""
    aligned = pd.DataFrame({"portfolio": returns, "benchmark": benchmark_returns}).dropna()
    if aligned.empty:
        raise ValueError("No overlapping observations for information ratio.")

    active = aligned["portfolio"] - aligned["benchmark"]
    tracking_error = active.std(ddof=1)
    if tracking_error == 0:
        return np.inf if active.mean() > 0 else -np.inf
    return active.mean() * math.sqrt(periods_per_year) / tracking_error


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Value at Risk using historical or parametric approach."""
    series = _validate_returns(returns)
    alpha_pct = (1 - confidence) * 100

    if method == "historical":
        return np.percentile(series, alpha_pct)
    if method == "parametric":
        return stats.norm.ppf(1 - confidence, loc=series.mean(), scale=series.std(ddof=1))
    raise ValueError("method must be 'historical' or 'parametric'")


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR (expected shortfall) based on historical losses."""
    series = _validate_returns(returns)
    var = value_at_risk(series, confidence, method="historical")
    tail_losses = series[series <= var]
    if tail_losses.empty:
        logger.warning("No tail losses below VaR; returning VaR value.")
        return var
    return tail_losses.mean()


def tail_ratio(returns: pd.Series) -> float:
    """Ratio of 95th percentile magnitude to 5th percentile magnitude."""
    series = _validate_returns(returns)
    right_tail = abs(np.percentile(series, 95))
    left_tail = abs(np.percentile(series, 5))
    if left_tail == 0:
        return np.inf
    return right_tail / left_tail


def skewness(returns: pd.Series) -> float:
    """Return skewness (third standardized moment)."""
    series = _validate_returns(returns, min_length=3)
    return stats.skew(series, bias=False)


def kurtosis(returns: pd.Series) -> float:
    """Return excess kurtosis (fourth central moment minus 3)."""
    series = _validate_returns(returns, min_length=4)
    return stats.kurtosis(series, bias=False)


def downside_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Average portfolio return divided by benchmark return when benchmark < 0."""
    aligned = pd.DataFrame({"portfolio": returns, "benchmark": benchmark_returns}).dropna()
    down = aligned[aligned["benchmark"] < 0]
    if down.empty:
        logger.warning("No down periods detected in benchmark.")
        return np.nan

    bench_mean = down["benchmark"].mean()
    if bench_mean == 0:
        return np.nan
    return down["portfolio"].mean() / bench_mean


def upside_capture(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Average portfolio return divided by benchmark return when benchmark > 0."""
    aligned = pd.DataFrame({"portfolio": returns, "benchmark": benchmark_returns}).dropna()
    up = aligned[aligned["benchmark"] > 0]
    if up.empty:
        logger.warning("No up periods detected in benchmark.")
        return np.nan

    bench_mean = up["benchmark"].mean()
    if bench_mean == 0:
        return np.nan
    return up["portfolio"].mean() / bench_mean


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Omega ratio comparing probability-weighted gains above a threshold vs losses below."""
    series = _validate_returns(returns)
    per_period_threshold = threshold / periods_per_year
    excess = series - per_period_threshold
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return np.inf
    return gains / losses


def rolling_volatility(
    returns: pd.Series,
    window: int = 21,
    periods_per_year: int = 252,
) -> pd.Series:
    """Rolling annualized volatility series for volatility targeting."""

    if window < 2:
        raise ValueError("window must be at least 2.")

    series = _validate_returns(returns, min_length=2)
    min_periods = min(window, len(series))
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=1)
    return rolling_std * math.sqrt(periods_per_year)


def compute_all_metrics(
    returns: pd.Series,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Return dictionary of core + advanced metrics."""
    metrics = {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
        "var_95": value_at_risk(returns, 0.95),
        "cvar_95": conditional_var(returns, 0.95),
        "tail_ratio": tail_ratio(returns),
        "skewness": skewness(returns),
        "kurtosis": kurtosis(returns),
        "omega": omega_ratio(returns, threshold=risk_free_rate, periods_per_year=periods_per_year),
    }

    rolling_window = max(2, min(21, len(_validate_returns(returns, min_length=2))))
    rolling_vol_series = rolling_volatility(
        returns,
        window=rolling_window,
        periods_per_year=periods_per_year,
    )
    metrics[f"rolling_vol_{rolling_window}"] = float(rolling_vol_series.iloc[-1])

    if benchmark_returns is not None:
        metrics["information_ratio"] = information_ratio(
            returns, benchmark_returns, periods_per_year
        )
        metrics["downside_capture"] = downside_capture(returns, benchmark_returns)
        metrics["upside_capture"] = upside_capture(returns, benchmark_returns)

    logger.info("Computed performance metrics: %s", metrics.keys())
    return metrics


if __name__ == "__main__":  # pragma: no cover
    np.random.seed(42)
    daily_returns = pd.Series(np.random.normal(0.0005, 0.01, 252))
    benchmark = pd.Series(np.random.normal(0.0004, 0.009, 252))

    metrics = compute_all_metrics(
        daily_returns,
        benchmark_returns=benchmark,
        risk_free_rate=0.03,
        periods_per_year=252,
    )

    logger.info("=== PERFORMANCE METRICS ===")
    for key, value in metrics.items():
        if "return" in key or "drawdown" in key or "var" in key:
            logger.info("%s: %.2f%%", key, value * 100)
        else:
            logger.info("%s: %.4f", key, value)
