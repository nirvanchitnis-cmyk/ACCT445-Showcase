"""Tests for src/analysis/panel_regression.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.panel_regression import (
    driscoll_kraay_regression,
    fama_macbeth_regression,
    fixed_effects_regression,
    prepare_panel_data,
    run_all_panel_regressions,
)
from src.utils.exceptions import DataValidationError


class TestPreparePanelData:
    """Tests for prepare_panel_data."""

    def test_prepare_creates_multiindex(self, sample_panel_data: pd.DataFrame) -> None:
        panel_df = prepare_panel_data(sample_panel_data)
        assert isinstance(panel_df.index, pd.MultiIndex)
        assert panel_df.index.names == ["ticker", "quarter"]
        assert len(panel_df) == len(sample_panel_data)

    def test_prepare_requires_columns(self, sample_panel_data: pd.DataFrame) -> None:
        trimmed = sample_panel_data.drop(columns=["ticker"])
        with pytest.raises(DataValidationError):
            prepare_panel_data(trimmed)


class TestFixedEffectsRegression:
    """Fixed-Effects regression tests."""

    def test_fe_runs(self, sample_panel_data: pd.DataFrame) -> None:
        panel_df = prepare_panel_data(sample_panel_data)
        results = fixed_effects_regression(panel_df, independent_vars=["CNOI", "log_mcap"])

        assert "coefficients" in results
        assert set(results["coefficients"]) == {"CNOI", "log_mcap"}
        assert results["effects"] == "twoway"

    def test_fe_detects_known_signal(self) -> None:
        np.random.seed(7)
        tickers = [f"STOCK{i:02d}" for i in range(15)]
        quarters = pd.period_range("2019Q1", periods=8, freq="Q")
        rows = []
        for ticker in tickers:
            for quarter in quarters:
                cnoi = 10 + np.random.uniform(-2, 2)
                ret = -0.004 * cnoi + np.random.normal(0, 0.005)
                rows.append({"ticker": ticker, "quarter": quarter, "CNOI": cnoi, "ret_fwd": ret})
        df = pd.DataFrame(rows)
        panel_df = prepare_panel_data(df)

        results = fixed_effects_regression(
            panel_df, independent_vars=["CNOI"], entity_effects=True, time_effects=False
        )

        assert results["coefficients"]["CNOI"] < 0
        assert abs(results["t_stats"]["CNOI"]) > 2


class TestFamaMacbethRegression:
    """Fama-MacBeth regression tests."""

    def test_fm_runs(self, sample_panel_data: pd.DataFrame) -> None:
        results = fama_macbeth_regression(sample_panel_data, independent_vars=["CNOI"])
        assert "coefficients" in results
        assert "CNOI" in results["coefficients"]
        assert results["n_periods"] > 5

    def test_fm_recovers_signal(self) -> None:
        np.random.seed(1)
        tickers = [f"STOCK{i:02d}" for i in range(20)]
        periods = pd.period_range("2020Q1", periods=10, freq="Q")
        rows = []
        for quarter in periods:
            for ticker in tickers:
                cnoi = np.random.uniform(5, 15)
                ret = -0.003 * cnoi + np.random.normal(0, 0.01)
                rows.append({"ticker": ticker, "quarter": quarter, "CNOI": cnoi, "ret_fwd": ret})
        df = pd.DataFrame(rows)

        results = fama_macbeth_regression(df, independent_vars=["CNOI"])

        assert results["coefficients"]["CNOI"] < -0.001
        assert abs(results["t_stats"]["CNOI"]) > 1.5


class TestDriscollKraayRegression:
    """Driscoll-Kraay regression tests."""

    def test_dk_runs(self, sample_panel_data: pd.DataFrame) -> None:
        panel_df = prepare_panel_data(sample_panel_data)
        results = driscoll_kraay_regression(panel_df, independent_vars=["CNOI"])

        assert "coefficients" in results
        assert "CNOI" in results["coefficients"]
        assert results["max_lags"] == 4


class TestRunAllPanelRegressions:
    """Integration tests for the comparison wrapper."""

    def test_all_methods_return_consistent_sign(self, sample_panel_data: pd.DataFrame) -> None:
        results = run_all_panel_regressions(sample_panel_data, independent_vars=["CNOI"])
        fe = results["FE"]["coefficients"]["CNOI"]
        fm = results["FM"]["coefficients"]["CNOI"]
        dk = results["DK"]["coefficients"]["CNOI"]

        assert np.sign(fe) == np.sign(dk)
        assert np.sign(fe) == np.sign(fm) or abs(fm) < 1e-3
