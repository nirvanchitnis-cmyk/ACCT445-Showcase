"""Tests for decile backtest module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.decile_backtest import (
    assign_deciles,
    compute_decile_returns,
    compute_long_short,
    newey_west_tstat,
    run_decile_backtest,
)
from src.utils.exceptions import DataValidationError


class TestAssignDeciles:
    """Tests for decile assignment helper."""

    def test_deciles_cover_range(self, sample_decile_data: pd.DataFrame):
        ranked = assign_deciles(sample_decile_data, "CNOI", n_groups=5)
        assert ranked["decile"].between(1, 5).all()

    def test_missing_score_column_raises(self, sample_decile_data: pd.DataFrame):
        with pytest.raises(DataValidationError):
            assign_deciles(sample_decile_data.drop(columns=["CNOI"]), "CNOI")

    def test_descending_order(self, sample_decile_data: pd.DataFrame):
        ranked = assign_deciles(sample_decile_data, "CNOI", ascending=False)
        top_decile = ranked[ranked["decile"] == 1]["CNOI"].mean()
        bottom_decile = ranked[ranked["decile"] == 10]["CNOI"].mean()
        assert top_decile > bottom_decile


class TestComputeDecileReturns:
    """Tests for decile aggregation."""

    def test_equal_weight_returns(self, sample_decile_data: pd.DataFrame):
        ranked = assign_deciles(sample_decile_data, "CNOI")
        decile_ret = compute_decile_returns(ranked, return_col="ret_fwd")
        assert {"decile", "date", "ret_fwd"}.issubset(decile_ret.columns)

    def test_missing_columns_raise(self, sample_decile_data: pd.DataFrame):
        with pytest.raises(DataValidationError):
            compute_decile_returns(sample_decile_data, decile_col="missing")

    def test_value_weight_returns(self, sample_decile_data: pd.DataFrame):
        ranked = assign_deciles(sample_decile_data, "CNOI")
        decile_ret = compute_decile_returns(
            ranked,
            return_col="ret_fwd",
            weight_col="market_cap",
        )
        assert not decile_ret["ret_fwd"].isna().all()


class TestNeweyWest:
    """Tests for HAC t-stat implementation."""

    def test_positive_trend_positive_tstat(self):
        np.random.seed(42)
        returns = np.random.normal(0.01, 0.02, 200)
        mean, se, t_stat = newey_west_tstat(returns, lags=2)
        assert mean > 0
        assert se > 0
        assert t_stat > 0

    def test_short_series_returns_nan(self):
        mean, se, t_stat = newey_west_tstat(np.array([0.01]))
        assert np.isnan(mean)
        assert np.isnan(se)
        assert np.isnan(t_stat)


class TestRunDecileBacktest:
    """Integration style tests for run_decile_backtest."""

    def test_backtest_outputs(self, sample_decile_data: pd.DataFrame):
        returns_df = sample_decile_data[["ticker", "date", "ret_fwd"]].rename(
            columns={"ret_fwd": "ret_fwd"}
        )
        summary, ls = run_decile_backtest(
            sample_decile_data,
            returns_df,
            score_col="CNOI",
            return_col="ret_fwd",
            n_deciles=10,
        )

        assert len(summary) == 10
        assert {"portfolio", "mean_ret"}.issubset(ls.columns)

    def test_long_short_alignment(self, sample_decile_data: pd.DataFrame):
        returns_df = sample_decile_data[["ticker", "date", "ret_fwd"]]
        summary, ls = run_decile_backtest(
            sample_decile_data,
            returns_df,
            score_col="CNOI",
            return_col="ret_fwd",
            n_deciles=5,
        )

        ls_mean = ls.loc[0, "mean_ret"]
        d1 = summary[summary["decile"] == 1]["mean_ret"].iloc[0]
        d5 = summary[summary["decile"] == 5]["mean_ret"].iloc[0]
        assert np.isclose(ls_mean, d1 - d5, atol=1e-6)

    def test_missing_required_columns(self, sample_decile_data: pd.DataFrame):
        trimmed = sample_decile_data.drop(columns=["ticker"])
        returns_df = sample_decile_data[["ticker", "date", "ret_fwd"]]
        with pytest.raises(DataValidationError):
            run_decile_backtest(
                trimmed,
                returns_df,
                score_col="CNOI",
                return_col="ret_fwd",
            )
