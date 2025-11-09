"""Tests for dashboard module helpers and rendering hooks."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from src.dashboard import app


def _mock_streamlit(monkeypatch):
    """Replace streamlit with a lightweight mock."""
    dummy = MagicMock()
    dummy.columns.side_effect = lambda n: [MagicMock() for _ in range(n)]
    dummy.sidebar = MagicMock()
    dummy.sidebar.radio.return_value = "Overview"
    dummy.sidebar.markdown.return_value = None
    dummy.sidebar.title.return_value = None
    dummy.sidebar.caption.return_value = None
    monkeypatch.setattr(app, "st", dummy)
    return dummy


def test_dashboard_imports():
    """Ensure the Streamlit app module imports cleanly."""
    assert hasattr(app, "st")


def test_load_results_callable():
    """load_results should be exposed for caching."""
    assert callable(app.load_results)


def test_prepare_returns_filters_missing():
    """_prepare_returns should drop NaNs and empty results."""
    df = pd.DataFrame({"ret": [0.01, None, -0.02]})
    series = app._prepare_returns(df)
    assert series is not None
    assert len(series) == 2
    assert app._prepare_returns(None) is None


def test_calculate_overview_metrics():
    """Overview metrics should include required keys."""
    returns = pd.Series([0.01, -0.005, 0.015])
    metrics = app._calculate_overview_metrics(returns, ann_factor=12)
    assert set(metrics.keys()) == {"ann_return", "sharpe", "drawdown", "win_rate"}


def test_compute_data_quality_handles_missing():
    """Data quality helper should summarize coverage."""
    df = pd.DataFrame(
        {
            "ticker": ["AAA", None, "CCC"],
            "cik": [1, 2, 3],
            "company_name": ["A", "B", "C"],
        }
    )
    stats = app._compute_data_quality(df)
    assert stats["missing"] == 1
    assert stats["total"] == 3
    assert 0 < stats["coverage"] < 100


def test_display_functions_execute(monkeypatch):
    """Core display functions should run without raising."""
    _mock_streamlit(monkeypatch)
    long_short = pd.DataFrame(
        {
            "period": pd.date_range("2023-01-31", periods=6, freq="ME"),
            "ret": [0.01, -0.005, 0.007, -0.002, 0.004, 0.006],
        }
    )
    decile_summary = pd.DataFrame(
        {"decile": range(1, 11), "mean_ret": [i * 0.001 for i in range(10)]}
    )
    event_summary = pd.DataFrame({"cnoi_quartile": ["Q1", "Q2"], "CAR_mean": [-0.05, -0.02]})
    event_daily = pd.DataFrame(
        {
            "date": pd.date_range("2023-03-01", periods=4, freq="D"),
            "cnoi_quartile": ["Q1", "Q1", "Q2", "Q2"],
            "cum_ar": [-0.02, -0.01, -0.03, -0.025],
        }
    )
    cnoi = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", None],
            "cik": [1, 2, 3],
            "company_name": ["A", "B", "C"],
            "CNOI": [10, 20, 30],
        }
    )
    data = {
        "long_short": long_short,
        "decile_summary": decile_summary,
        "event_study": event_summary,
        "event_study_daily": event_daily,
        "cnoi_tickers": cnoi,
    }
    app._display_overview(data)
    app._display_decile(data)
    app._display_event_study(data)
    app._display_risk_metrics(data)
    app._display_data_quality(data)
