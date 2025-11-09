"""Tests for transaction cost modeling utilities."""

import numpy as np
import pandas as pd
import pytest

from src.utils.exceptions import DataValidationError
from src.utils.transaction_costs import (
    BPS,
    apply_transaction_costs,
    compute_market_impact,
    estimate_bid_ask_spread,
    estimate_slippage,
)


@pytest.fixture
def sample_series():
    """Volatility/market cap series with aligned index."""

    volatility = pd.Series(
        [0.25, 0.30, 0.18],
        index=["A", "B", "C"],
        name="vol",
    )
    market_cap = pd.Series(
        [5e9, 15e9, 50e9],
        index=["A", "B", "C"],
        name="mcap",
    )
    return volatility, market_cap


@pytest.fixture
def sample_returns():
    """Quarterly return series for cost application tests."""

    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4, freq="QE-DEC"),
            "ret": [0.03, 0.01, -0.005, 0.02],
        }
    )


def test_estimate_bid_ask_spread_behaves(sample_series):
    """Large caps should have tighter spreads than small caps."""

    volatility, market_cap = sample_series
    spreads = estimate_bid_ask_spread(volatility, market_cap)

    assert spreads.between(1.0, 50.0).all()
    assert spreads.loc["C"] < spreads.loc["A"]


def test_estimate_bid_ask_spread_requires_overlap():
    """Function should raise when there is no overlapping index."""

    vol = pd.Series([0.2], index=["A"])
    cap = pd.Series([1e9], index=["B"])

    with pytest.raises(DataValidationError):
        estimate_bid_ask_spread(vol, cap)


def test_compute_market_impact_scales_with_trade_size():
    """Larger trades relative to ADV should incur higher impact."""

    small_trade = compute_market_impact(
        trade_value=1e5,
        avg_daily_volume=1e7,
        volatility=0.2,
    )
    large_trade = compute_market_impact(
        trade_value=5e6,
        avg_daily_volume=1e7,
        volatility=0.2,
    )

    assert 0 <= small_trade < large_trade <= 100


def test_compute_market_impact_handles_illiquid_names():
    """When ADV is zero the function should cap impact at 100 bps."""

    assert compute_market_impact(1e6, avg_daily_volume=0) == 100.0


def test_estimate_slippage_urgency_levels(sample_series):
    """Slippage should increase with execution urgency."""

    volatility, _ = sample_series
    spreads = estimate_bid_ask_spread(volatility, volatility * 0 + 5e9)
    low = estimate_slippage(spreads, urgency="low")
    high = estimate_slippage(spreads, urgency="high")

    assert np.all(high > low)


def test_apply_transaction_costs_reduces_returns(sample_returns):
    """Net returns should reflect the expected cost drag."""

    turnover = 0.5
    result = apply_transaction_costs(
        sample_returns,
        turnover=turnover,
        avg_spread_bps=4.0,
        commission_bps=1.0,
        avg_impact_bps=2.5,
        slippage_bps=0.5,
    )

    total_cost_bps = (4.0 / 2) + 1.0 + 2.5 + 0.5
    expected_drag = turnover * total_cost_bps * BPS

    np.testing.assert_allclose(
        result["transaction_cost"].to_numpy(),
        np.full(len(sample_returns), expected_drag),
    )
    np.testing.assert_allclose(
        result["net_ret"],
        sample_returns["ret"] - expected_drag,
    )


def test_apply_transaction_costs_missing_column_raises(sample_returns):
    """Missing return column should raise validation error."""

    df = sample_returns.rename(columns={"ret": "return"})
    with pytest.raises(DataValidationError):
        apply_transaction_costs(df, turnover=0.5)
