import numpy as np
import pandas as pd
import pytest

from src.utils import delisting_returns


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    base = np.linspace(100, 110, num=len(dates))
    noise = np.linspace(1, -1, num=len(dates))
    return pd.DataFrame(
        {
            "KEEP": base + noise,
            "DROP": base - noise,
        },
        index=dates,
    )


def test_apply_delisting_returns_applies_penalty_and_blanks_future_rows():
    prices = _sample_prices()
    delist_dates = {
        "DROP": prices.index[3].strftime("%Y-%m-%d"),
        "MISSING": prices.index[2].strftime("%Y-%m-%d"),
    }

    returns, stats = delisting_returns.apply_delisting_returns(prices, delist_dates, penalty=-0.4)

    target_date = prices.index[3]
    assert pytest.approx(-0.4) == returns.loc[target_date, "DROP"]
    assert np.isnan(returns.loc[prices.index[4], "DROP"])
    assert stats["n_tickers"] == 1
    assert stats["n_obs_replaced"] >= 1


def test_apply_delisting_returns_handles_future_delist_date():
    prices = _sample_prices()
    future_date = (prices.index[-1] + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    returns, stats = delisting_returns.apply_delisting_returns(
        prices,
        {"DROP": future_date},
        penalty=-0.5,
    )

    # No adjustments should be applied when delist date exceeds data range
    assert stats["n_tickers"] == 0
    assert returns.equals(prices.pct_change())


def test_apply_delisting_returns_forward_fills_to_next_available_date():
    prices = _sample_prices()
    mid_day = (prices.index[1] + pd.Timedelta(hours=12)).isoformat()
    returns, stats = delisting_returns.apply_delisting_returns(
        prices,
        {"DROP": mid_day},
        penalty=-0.25,
    )

    # Penalty should apply on the next trading day because midday timestamp was not exact
    assert pytest.approx(-0.25) == returns.loc[prices.index[2], "DROP"]
    assert stats["n_tickers"] == 1


def test_apply_delisting_returns_requires_dataframe():
    with pytest.raises(TypeError):
        delisting_returns.apply_delisting_returns([], {})


def test_apply_delisting_returns_with_empty_mapping_returns_pct_change():
    prices = _sample_prices()
    result = delisting_returns.apply_delisting_returns(prices, {})
    pd.testing.assert_frame_equal(result, prices.pct_change())


def test_estimate_delisting_sensitivity_runs_multiple_penalties():
    prices = _sample_prices()
    delist_dates = {"DROP": prices.index[2].strftime("%Y-%m-%d")}

    penalties = [-0.1, -0.3, -0.5]
    summary = delisting_returns.estimate_delisting_sensitivity(
        prices, delist_dates, penalties=penalties
    )

    assert summary["penalty"].tolist() == penalties
    assert {"mean_return", "std_return", "sharpe", "n_obs_replaced", "pct_replaced"}.issubset(
        summary.columns
    )


def test_estimate_delisting_sensitivity_defaults_to_standard_penalties():
    prices = _sample_prices()
    delist_dates = {"DROP": prices.index[2].strftime("%Y-%m-%d")}

    summary = delisting_returns.estimate_delisting_sensitivity(prices, delist_dates)
    # Default penalties list contains four scenarios
    assert len(summary) == 4
