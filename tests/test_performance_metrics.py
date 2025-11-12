"""Tests for src/utils/performance_metrics.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import src.utils.performance_metrics as perf
from src.utils.performance_metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_all_metrics,
    conditional_var,
    downside_capture,
    information_ratio,
    kurtosis,
    max_drawdown,
    omega_ratio,
    rolling_volatility,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tail_ratio,
    upside_capture,
    value_at_risk,
)


@pytest.fixture
def normal_returns() -> pd.Series:
    """Simulated daily returns resembling a normal distribution."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 252))


@pytest.fixture
def skewed_returns() -> pd.Series:
    """Right-skewed return distribution."""
    np.random.seed(7)
    return pd.Series(stats.skewnorm.rvs(a=5, loc=0.001, scale=0.02, size=252))


@pytest.fixture
def benchmark_returns() -> pd.Series:
    """Simulated benchmark returns."""
    np.random.seed(21)
    return pd.Series(np.random.normal(0.0008, 0.015, 252))


class TestCoreMetrics:
    """Smoke tests for baseline metrics."""

    def test_annualized_return_positive(self, normal_returns: pd.Series) -> None:
        ann = annualized_return(normal_returns, periods_per_year=252)
        assert np.isfinite(ann)

    def test_annualized_volatility(self, normal_returns: pd.Series) -> None:
        vol = annualized_volatility(normal_returns, periods_per_year=252)
        assert vol > 0

    def test_sharpe_ratio_sign(self, normal_returns: pd.Series) -> None:
        ratio = sharpe_ratio(normal_returns, risk_free_rate=0.02)
        assert np.isfinite(ratio)

    def test_sortino_ratio(self, normal_returns: pd.Series) -> None:
        ratio = sortino_ratio(normal_returns, risk_free_rate=0.02)
        assert np.isfinite(ratio)

    def test_max_drawdown_negative(self, normal_returns: pd.Series) -> None:
        mdd = max_drawdown(normal_returns)
        assert mdd <= 0

    def test_calmar_ratio(self, normal_returns: pd.Series) -> None:
        calmar = calmar_ratio(normal_returns)
        assert np.isfinite(calmar)

    def test_information_ratio(
        self, normal_returns: pd.Series, benchmark_returns: pd.Series
    ) -> None:
        ir = information_ratio(normal_returns, benchmark_returns)
        assert np.isfinite(ir)


class TestRiskMetrics:
    """VaR / CVaR / tail ratio tests."""

    def test_value_at_risk_ordering(self, normal_returns: pd.Series) -> None:
        var_90 = value_at_risk(normal_returns, confidence=0.90)
        var_95 = value_at_risk(normal_returns, confidence=0.95)
        var_99 = value_at_risk(normal_returns, confidence=0.99)
        assert var_90 > var_95 > var_99

    def test_parametric_vs_historical_var(self, normal_returns: pd.Series) -> None:
        hist = value_at_risk(normal_returns, method="historical")
        param = value_at_risk(normal_returns, method="parametric")
        assert np.isfinite(hist)
        assert np.isfinite(param)

    def test_conditional_var_more_extreme(self, normal_returns: pd.Series) -> None:
        var = value_at_risk(normal_returns)
        cvar = conditional_var(normal_returns)
        assert cvar <= var

    def test_tail_ratio_behavior(
        self, skewed_returns: pd.Series, normal_returns: pd.Series
    ) -> None:
        tr_skewed = tail_ratio(skewed_returns)
        tr_normal = tail_ratio(normal_returns)
        assert tr_skewed > tr_normal


class TestDistributionMetrics:
    """Skewness and kurtosis coverage."""

    def test_skewness_signs(self, skewed_returns: pd.Series, normal_returns: pd.Series) -> None:
        assert skewness(skewed_returns) > 0
        assert abs(skewness(normal_returns)) < 1

    def test_kurtosis_bounds(self, normal_returns: pd.Series) -> None:
        kurt = kurtosis(normal_returns)
        assert -1 < kurt < 1


class TestCaptureRatios:
    """Upside/Downside capture tests."""

    def test_capture_ratios_known_outperformance(self) -> None:
        benchmark = pd.Series([0.01, -0.01, 0.02, -0.02])
        portfolio = benchmark * 2
        assert 1.9 < upside_capture(portfolio, benchmark) < 2.1
        assert 1.9 < downside_capture(portfolio, benchmark) < 2.1

    def test_capture_handles_missing_periods(self) -> None:
        benchmark = pd.Series([0.01, 0.02, 0.03])
        portfolio = pd.Series([0.02, 0.03, 0.04])
        uc = upside_capture(portfolio, benchmark)
        assert np.isfinite(uc)


class TestOmegaRatio:
    """Omega ratio behavior tests."""

    def test_omega_above_one_for_positive_returns(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.015, 0.012])
        assert omega_ratio(returns) > 1.0

    def test_omega_below_one_for_negative_returns(self) -> None:
        returns = pd.Series([-0.01, -0.015, -0.02])
        assert omega_ratio(returns) < 1.0


class TestRollingVolatility:
    """Rolling volatility behavior."""

    def test_rolling_volatility_matches_manual(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.015, 0.0, 0.01])
        window = 3
        result = rolling_volatility(returns, window=window, periods_per_year=252)
        manual = returns.rolling(window=window, min_periods=window).std(ddof=1) * math.sqrt(252)
        pd.testing.assert_series_equal(result, manual)

    def test_rolling_volatility_handles_short_series(self) -> None:
        returns = pd.Series([0.01, -0.02, 0.015])
        series = rolling_volatility(returns, window=5, periods_per_year=252)
        assert np.isfinite(series.iloc[-1])

    def test_rolling_volatility_invalid_window(self) -> None:
        returns = pd.Series([0.01, 0.02])
        with pytest.raises(ValueError):
            rolling_volatility(returns, window=1)


class TestComputeAllMetrics:
    """Integration tests for compute_all_metrics."""

    def test_compute_without_benchmark(self, normal_returns: pd.Series) -> None:
        metrics = compute_all_metrics(normal_returns, risk_free_rate=0.02)
        required_keys = {
            "ann_return",
            "ann_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "var_95",
            "cvar_95",
            "tail_ratio",
            "skewness",
            "kurtosis",
            "omega",
            "rolling_vol_21",
        }
        assert required_keys.issubset(metrics.keys())

    def test_compute_with_benchmark(
        self,
        normal_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> None:
        metrics = compute_all_metrics(
            normal_returns,
            benchmark_returns=benchmark_returns,
            risk_free_rate=0.02,
        )
        for key in ("information_ratio", "downside_capture", "upside_capture"):
            assert key in metrics


class TestEdgeCases:
    """Edge-case coverage for branch-heavy helpers."""

    def test_validate_returns_enforces_length(self):
        with pytest.raises(ValueError):
            perf.annualized_volatility(pd.Series([0.01]))

    def test_annualized_return_handles_non_positive_growth(self):
        returns = pd.Series([-1.0, 0.0, 0.01])
        approx = perf.annualized_return(returns, periods_per_year=1)
        assert approx == pytest.approx(returns.mean())

    def test_sharpe_ratio_zero_volatility(self):
        series = pd.Series([0.01, 0.01, 0.01])
        assert perf.sharpe_ratio(series) == np.inf

    def test_sortino_ratio_without_downside(self):
        series = pd.Series([0.02, 0.03, 0.025])
        assert perf.sortino_ratio(series) == np.inf

    def test_information_ratio_requires_overlap(self):
        portfolio = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2, freq="D"))
        benchmark = pd.Series([0.01, 0.02], index=pd.date_range("2025-01-01", periods=2, freq="D"))
        with pytest.raises(ValueError):
            perf.information_ratio(portfolio, benchmark)

    def test_information_ratio_zero_tracking_error(self):
        returns = pd.Series([0.01, 0.02], index=pd.date_range("2024-01-01", periods=2))
        assert perf.information_ratio(returns, returns) == -np.inf

    def test_value_at_risk_rejects_unknown_method(self, normal_returns):
        with pytest.raises(ValueError):
            perf.value_at_risk(normal_returns, method="gaussian")

    def test_conditional_var_handles_empty_tail(self, monkeypatch):
        returns = pd.Series([0.01, 0.02, 0.03])
        monkeypatch.setattr(perf, "value_at_risk", lambda *_, **__: -1.0)
        assert perf.conditional_var(returns) == -1.0

    def test_tail_ratio_returns_infinite_when_left_tail_zero(self):
        returns = pd.Series([0.0, 0.0, 0.1, 0.2, 0.3])
        assert perf.tail_ratio(returns) == np.inf

    def test_downside_capture_no_down_periods(self):
        benchmark = pd.Series([0.01, 0.02, 0.03])
        portfolio = pd.Series([0.02, 0.03, 0.04])
        assert np.isnan(perf.downside_capture(portfolio, benchmark))

    def test_upside_capture_no_up_periods(self):
        benchmark = pd.Series([-0.01, -0.02, -0.03])
        portfolio = pd.Series([-0.01, -0.02, -0.03])
        assert np.isnan(perf.upside_capture(portfolio, benchmark))

    def test_omega_ratio_infinite_without_losses(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        assert perf.omega_ratio(returns) == np.inf
