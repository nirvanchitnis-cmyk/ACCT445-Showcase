"""Tests for the event study pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

import src.analysis.event_study as event_study
from src.utils.exceptions import DataValidationError


class TestComputeMarketModelParams:
    """Tests for market model estimation."""

    def test_successful_estimation(self, sample_event_study_returns, sample_market_returns):
        params = event_study.compute_market_model_params(
            sample_event_study_returns,
            sample_market_returns,
            estimation_window_start="2023-01-01",
            estimation_window_end="2023-02-28",
        )
        assert {"ticker", "alpha", "beta"}.issubset(params.columns)
        assert len(params) > 0

    def test_insufficient_data(self):
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        returns = pd.DataFrame(
            {"date": dates, "ticker": ["BAC"] * 3, "return": [0.01, 0.02, 0.015]}
        )
        market = pd.Series([0.01, 0.02, 0.0], index=dates)

        with pytest.raises(DataValidationError):
            event_study.compute_market_model_params(
                returns,
                market,
                estimation_window_start="2023-01-01",
                estimation_window_end="2023-01-05",
            )

    def test_invalid_market_index(self, sample_event_study_returns):
        market = pd.Series([0.01, 0.02, 0.03], index=[0, 1, 2])
        with pytest.raises(DataValidationError):
            event_study.compute_market_model_params(
                sample_event_study_returns,
                market,
                estimation_window_start="2023-01-01",
                estimation_window_end="2023-01-05",
            )


class TestAbnormalReturns:
    """Tests for abnormal return calculations."""

    def test_abnormal_returns_shape(self, sample_event_study_returns, sample_market_returns):
        params = event_study.compute_market_model_params(
            sample_event_study_returns,
            sample_market_returns,
            estimation_window_start="2023-01-01",
            estimation_window_end="2023-02-15",
        )

        ar = event_study.compute_abnormal_returns(
            sample_event_study_returns,
            sample_market_returns,
            params,
            event_start="2023-03-01",
            event_end="2023-03-10",
        )

        assert {"ticker", "abnormal_return", "expected_return"}.issubset(ar.columns)
        assert len(ar) > 0


class TestCumulativeAbnormalReturns:
    """Tests for CAR aggregation."""

    def test_car_summary(self, sample_event_study_returns, sample_market_returns):
        params = event_study.compute_market_model_params(
            sample_event_study_returns,
            sample_market_returns,
            estimation_window_start="2023-01-01",
            estimation_window_end="2023-02-15",
        )
        ar = event_study.compute_abnormal_returns(
            sample_event_study_returns,
            sample_market_returns,
            params,
            event_start="2023-03-01",
            event_end="2023-03-10",
        )
        car = event_study.compute_cumulative_abnormal_returns(ar)
        assert {"ticker", "CAR", "n_days"}.issubset(car.columns)
        assert (car["n_days"] > 0).all()


class TestRunEventStudy:
    """End-to-end pipeline tests."""

    def test_full_pipeline(self, sample_event_study_returns, sample_market_returns):
        cnoi = pd.DataFrame(
            {
                "ticker": sample_event_study_returns["ticker"].unique(),
                "filing_date": pd.to_datetime("2023-02-01"),
                "CNOI": [10 + i for i in range(len(sample_event_study_returns["ticker"].unique()))],
            }
        )

        summary, car = event_study.run_event_study(
            sample_event_study_returns,
            sample_market_returns,
            cnoi,
            estimation_start="2023-01-01",
            estimation_end="2023-02-28",
            event_start="2023-03-01",
            event_end="2023-03-15",
            pre_event_cutoff="2023-02-15",
        )

        assert not summary.empty
        assert not car.empty

    def test_quartile_relationship(self):
        car = pd.DataFrame(
            {
                "ticker": list("ABCDEFGH"),
                "CAR": [0.01, -0.02, 0.03, -0.01, 0.015, -0.03, 0.025, -0.005],
            }
        )
        cnoi = pd.DataFrame(
            {
                "ticker": list("ABCDEFGH"),
                "filing_date": pd.to_datetime(["2023-01-01"] * 8),
                "CNOI": [5, 7, 12, 15, 18, 20, 25, 30],
            }
        )

        summary = event_study.test_cnoi_car_relationship(
            car,
            cnoi,
            pre_event_cutoff="2023-02-01",
        )

        assert not summary.empty
