"""Tests for src/analysis/robustness.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.robustness import (
    bootstrap_backtest,
    monte_carlo_long_short,
    permutation_test,
    subsample_analysis,
)
from src.utils.exceptions import DataValidationError


def _build_panel(signal: bool, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tickers = [f"BK{i:02d}" for i in range(30)]
    periods = pd.period_range("2021Q1", periods=12, freq="Q")

    rows: list[dict[str, float]] = []
    for period in periods:
        for ticker in tickers:
            cnoi = rng.uniform(10, 35)
            noise = rng.normal(0, 0.02)
            slope = -0.0025 if signal else 0.0
            ret = slope * (cnoi - 20) + noise
            rows.append(
                {
                    "ticker": ticker,
                    "date": period.to_timestamp(),
                    "CNOI": cnoi,
                    "ret_fwd": ret,
                    "regime": "pre" if period.year <= 2022 else "post",
                    "market_cap": rng.uniform(1e9, 5e9),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def signal_panel() -> pd.DataFrame:
    return _build_panel(signal=True, seed=7)


@pytest.fixture
def null_panel() -> pd.DataFrame:
    return _build_panel(signal=False, seed=13)


def test_bootstrap_backtest_produces_confidence_interval(signal_panel: pd.DataFrame) -> None:
    result = bootstrap_backtest(
        signal_panel,
        "CNOI",
        n_bootstrap=50,
        random_seed=1,
        progress=False,
    )
    assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]
    assert result["n_draws"] > 10


def test_bootstrap_supports_value_weighting(signal_panel: pd.DataFrame) -> None:
    result = bootstrap_backtest(
        signal_panel,
        "CNOI",
        weight_col="market_cap",
        n_bootstrap=10,
        random_seed=0,
        progress=False,
    )
    assert result["n_draws"] > 0


def test_bootstrap_requires_positive_iterations(signal_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        bootstrap_backtest(signal_panel, "CNOI", n_bootstrap=0, progress=False)


def test_permutation_detects_signal(signal_panel: pd.DataFrame) -> None:
    result = permutation_test(
        signal_panel,
        "CNOI",
        n_permutations=50,
        random_seed=2,
        progress=False,
    )
    assert result["p_value"] < 0.2


def test_permutation_high_p_value_without_signal(null_panel: pd.DataFrame) -> None:
    result = permutation_test(
        null_panel,
        "CNOI",
        n_permutations=50,
        random_seed=3,
        progress=False,
    )
    assert result["p_value"] > 0.05


def test_permutation_requires_positive_iterations(signal_panel: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        permutation_test(signal_panel, "CNOI", n_permutations=0, progress=False)


def test_subsample_analysis_returns_each_regime(signal_panel: pd.DataFrame) -> None:
    summary = subsample_analysis(
        signal_panel,
        "CNOI",
        split_col="regime",
        split_values=["pre", "post"],
        min_obs=100,
    )
    assert set(summary["regime"]) == {"pre", "post"}
    assert summary["long_short_ret"].notna().all()


def test_subsample_analysis_raises_when_all_skipped(signal_panel: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        subsample_analysis(
            signal_panel,
            "CNOI",
            split_col="regime",
            split_values=["pre"],
            min_obs=1000,
        )


def test_monte_carlo_long_short_distribution() -> None:
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(0.01, 0.005, 50))
    result = monte_carlo_long_short(returns, horizon=6, n_scenarios=200, random_seed=4)
    assert np.isfinite(result["mean"])
    assert result["p05"] <= result["mean"] <= result["p95"]
    assert len(result["distribution"]) == 200


def test_monte_carlo_input_validation() -> None:
    rng = np.random.default_rng(6)
    returns = pd.Series(rng.normal(0.0, 0.01, 10))
    with pytest.raises(ValueError):
        monte_carlo_long_short(returns, horizon=0)
    with pytest.raises(ValueError):
        monte_carlo_long_short(returns, n_scenarios=0)
    with pytest.raises(DataValidationError):
        monte_carlo_long_short(pd.Series(dtype=float))
