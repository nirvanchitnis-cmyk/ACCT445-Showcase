"""Transaction cost modeling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

BPS = 1e-4  # Basis point expressed as decimal


def estimate_bid_ask_spread(
    volatility: pd.Series,
    market_cap: pd.Series,
    base_spread_bps: float = 5.0,
    min_spread_bps: float = 1.0,
    max_spread_bps: float = 50.0,
) -> pd.Series:
    """Estimate bid-ask spread as a function of volatility and firm size."""

    if volatility.empty or market_cap.empty:
        raise DataValidationError("Volatility and market_cap series must be non-empty.")

    volatility, market_cap = volatility.align(market_cap, join="inner")
    if volatility.empty:
        raise DataValidationError("Volatility and market_cap must share at least one index.")

    vol = volatility.clip(lower=0.0).astype(float)
    cap = market_cap.clip(lower=1e8).astype(float)

    mcap_billions = cap / 1e9
    spread = base_spread_bps * (1 + vol) * (1 / np.sqrt(mcap_billions))
    spread = spread.clip(lower=min_spread_bps, upper=max_spread_bps)

    logger.debug(
        "Estimated bid-ask spread: mean %.2f bps (range %.2f-%.2f)",
        spread.mean(),
        spread.min(),
        spread.max(),
    )
    return spread


def compute_market_impact(
    trade_value: float,
    avg_daily_volume: float,
    volatility: float = 0.2,
    impact_coefficient: float = 0.1,
) -> float:
    """Compute market impact cost using an Almgren-Chriss style approximation."""

    if trade_value <= 0:
        raise DataValidationError("trade_value must be positive.")

    if avg_daily_volume <= 0:
        logger.warning("avg_daily_volume <= 0 encountered; returning 100 bps impact.")
        return 100.0

    participation_rate = trade_value / avg_daily_volume
    participation_rate = max(participation_rate, 0.0)
    vol = max(volatility, 0.01)

    impact_bps = impact_coefficient * np.sqrt(participation_rate) * vol * 10000
    impact_bps = float(np.clip(impact_bps, 0.0, 100.0))

    logger.debug(
        "Market impact: trade_value=%s ADV=%s vol=%.2f -> %.2f bps",
        trade_value,
        avg_daily_volume,
        vol,
        impact_bps,
    )
    return impact_bps


def estimate_slippage(
    spread_bps: pd.Series | float,
    urgency: str = "normal",
) -> pd.Series | float:
    """Estimate slippage as a fraction of the prevailing spread."""

    urgency_map = {
        "low": 0.25,
        "normal": 0.5,
        "high": 0.85,
    }

    if urgency not in urgency_map:
        raise ValueError("urgency must be one of {'low', 'normal', 'high'}.")

    multiplier = urgency_map[urgency]

    if isinstance(spread_bps, pd.Series):
        return (spread_bps.astype(float) * multiplier).clip(lower=0.0)

    return float(max(spread_bps, 0.0) * multiplier)


def apply_transaction_costs(
    backtest_returns: pd.DataFrame,
    turnover: float,
    avg_spread_bps: float = 5.0,
    commission_bps: float = 1.0,
    avg_impact_bps: float = 2.0,
    slippage_bps: float | None = None,
    return_col: str = "ret",
) -> pd.DataFrame:
    """Apply transaction cost drag to backtest returns."""

    if return_col not in backtest_returns.columns:
        raise DataValidationError(f"Input data missing '{return_col}' column.")

    if turnover < 0:
        raise DataValidationError("turnover must be non-negative.")

    spread_component = max(avg_spread_bps, 0.0) / 2
    slippage_component = max(slippage_bps or 0.0, 0.0)

    total_cost_bps = spread_component + commission_bps + avg_impact_bps + slippage_component
    period_cost = turnover * total_cost_bps * BPS

    result = backtest_returns.copy()
    result["gross_ret"] = result[return_col]
    result["transaction_cost"] = period_cost
    result["net_ret"] = result["gross_ret"] - period_cost

    logger.info(
        "Applied transaction costs: turnover=%.2f, total cost %.2f bps (%.4f per period).",
        turnover,
        total_cost_bps,
        period_cost,
    )

    return result


__all__ = [
    "estimate_bid_ask_spread",
    "compute_market_impact",
    "estimate_slippage",
    "apply_transaction_costs",
    "BPS",
]
