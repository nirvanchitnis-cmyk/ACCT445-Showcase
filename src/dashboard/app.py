"""Interactive monitoring dashboard for the ACCT445 Showcase system."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.config import get_config_value
from src.utils.performance_metrics import (
    conditional_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)

RESULTS_DIR = Path(get_config_value("data.results_dir", "results"))

st.set_page_config(page_title="ACCT445 Showcase", layout="wide", page_icon="📊")


@st.cache_data(show_spinner=False)
def load_results() -> dict[str, pd.DataFrame | None]:
    """Best-effort loader for downstream result files."""

    def _first_existing(candidates: list[str]) -> pd.DataFrame | None:
        for filename in candidates:
            path = RESULTS_DIR / filename
            if path.exists():
                return pd.read_csv(path)
        return None

    return {
        "decile_summary": _first_existing(
            ["decile_summary_latest.csv", "decile_summary_lag2_equal.csv"]
        ),
        "long_short": _first_existing(
            ["decile_long_short_latest.csv", "decile_long_short_lag2_equal.csv"]
        ),
        "cnoi_tickers": _first_existing(["cnoi_with_tickers.csv"]),
        "event_study": _first_existing(["event_study_results.csv"]),
        "event_study_daily": _first_existing(["event_study_daily_cum_ar.csv"]),
    }


def _prepare_returns(long_short: pd.DataFrame | None) -> pd.Series | None:
    """Return cleaned long-short series for downstream metrics."""
    if long_short is None or "ret" not in long_short.columns:
        return None
    series = pd.Series(long_short["ret"]).dropna()
    return series if not series.empty else None


def _calculate_overview_metrics(returns: pd.Series, ann_factor: int) -> dict[str, float]:
    """Compute headline metrics for overview cards."""
    return {
        "ann_return": returns.mean() * ann_factor * 100,
        "sharpe": sharpe_ratio(returns, periods_per_year=ann_factor),
        "drawdown": max_drawdown(returns) * 100,
        "win_rate": (returns > 0).mean() * 100,
    }


def _compute_data_quality(
    cnoi_df: pd.DataFrame | None,
) -> dict[str, float] | None:
    if cnoi_df is None or cnoi_df.empty:
        return None
    total = len(cnoi_df)
    missing = cnoi_df["ticker"].isna().sum()
    coverage = (1 - missing / total) * 100 if total else 0.0
    return {"total": total, "missing": missing, "coverage": coverage}


def _display_overview(data: dict[str, pd.DataFrame | None]) -> None:
    st.title("📊 ACCT445 Showcase Overview")
    st.markdown("Monitoring bank disclosure opacity signals in real time.")

    returns_series = _prepare_returns(data.get("long_short"))
    long_short = data.get("long_short")

    if returns_series is not None and not returns_series.empty:
        ann_factor = get_config_value("backtest.periods_per_year", 12)
        metrics = _calculate_overview_metrics(returns_series, ann_factor)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Long-Short Return (Ann.)", f"{metrics['ann_return']:.2f}%")
        with col2:
            st.metric("Sharpe Ratio", f"{metrics['sharpe']:.2f}")
        with col3:
            st.metric("Max Drawdown", f"{metrics['drawdown']:.2f}%")
        with col4:
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")

        st.subheader("Cumulative Long-Short Returns")
        cum_returns = (1 + returns_series).cumprod() - 1
        fig = px.line(
            x=cum_returns.index,
            y=cum_returns.values,
            labels={"x": "Period", "y": "Cumulative Return"},
        )
        st.plotly_chart(fig, use_container_width=True)
    elif long_short is not None and "mean_ret" in long_short.columns:
        st.info(
            "Displaying summary statistics only. "
            "Run the Phase 5 runner to capture time-series returns."
        )
        st.dataframe(long_short, use_container_width=True)
    else:
        st.warning("No long-short results available. Run the backtest runner first.")

    st.markdown("---")
    st.subheader("System Status")
    status1, status2, status3 = st.columns(3)
    latest_files = list(RESULTS_DIR.glob("*latest*.csv"))
    latest_time = (
        max((f.stat().st_mtime for f in latest_files), default=None) if latest_files else None
    )
    with status1:
        st.metric("Results Directory", str(RESULTS_DIR.resolve()))
    with status2:
        if latest_time:
            timestamp = datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d %H:%M")
        else:
            timestamp = "N/A"
        st.metric("Last Update", timestamp)
    with status3:
        ticker_df = data.get("cnoi_tickers")
        total = len(ticker_df) if ticker_df is not None else 0
        st.metric("Tickers Tracked", total if total else "N/A")


def _display_decile(data: dict[str, pd.DataFrame | None]) -> None:
    st.title("📈 Decile Backtest Results")
    summary = data.get("decile_summary")
    if summary is None:
        st.warning("No decile summary available. Run the runner first.")
        return

    st.dataframe(summary, use_container_width=True)

    if {"decile", "mean_ret"}.issubset(summary.columns):
        st.subheader("Mean Return by Decile")
        fig = go.Figure(
            data=[
                go.Bar(
                    x=summary["decile"],
                    y=summary["mean_ret"],
                    marker_color="steelblue",
                )
            ]
        )
        fig.update_layout(xaxis_title="Decile", yaxis_title="Mean Return")
        st.plotly_chart(fig, use_container_width=True)


def _display_event_study(data: dict[str, pd.DataFrame | None]) -> None:
    st.title("📅 Event Study Results")
    summary = data.get("event_study")
    daily = data.get("event_study_daily")
    if summary is None:
        st.warning("No event study outputs available.")
        return

    st.dataframe(summary, use_container_width=True)
    if {"cnoi_quartile", "CAR_mean"}.issubset(summary.columns):
        st.subheader("Average CAR by Quartile")
        fig = px.bar(summary, x="cnoi_quartile", y="CAR_mean", color="cnoi_quartile")
        st.plotly_chart(fig, use_container_width=True)

    if daily is not None and {"date", "cum_ar", "cnoi_quartile"}.issubset(daily.columns):
        st.subheader("Daily Cumulative Abnormal Returns")
        fig = px.line(daily, x="date", y="cum_ar", color="cnoi_quartile")
        st.plotly_chart(fig, use_container_width=True)


def _display_risk_metrics(data: dict[str, pd.DataFrame | None]) -> None:
    st.title("⚠️ Risk Metrics")
    returns = _prepare_returns(data.get("long_short"))
    if returns is None:
        st.warning("No long-short data available.")
        return

    col1, col2 = st.columns(2)
    with col1:
        var_95 = value_at_risk(returns, confidence=0.95) * 100
        cvar_95 = conditional_var(returns, confidence=0.95) * 100
        st.metric("95% VaR", f"{var_95:.2f}%")
        st.metric("95% CVaR", f"{cvar_95:.2f}%")
        st.metric(
            "Sharpe Ratio",
            f"{sharpe_ratio(returns, periods_per_year=12):.2f}",
        )
    with col2:
        st.metric("Sortino Ratio", f"{sortino_ratio(returns, periods_per_year=12):.2f}")
        volatility = returns.std(ddof=1) * np.sqrt(12) * 100
        st.metric("Volatility (Ann.)", f"{volatility:.2f}%")
        st.metric("Max Drawdown", f"{max_drawdown(returns) * 100:.2f}%")

    st.subheader("Return Distribution")
    fig = px.histogram(returns, nbins=30, title="Distribution of Long-Short Returns")
    st.plotly_chart(fig, use_container_width=True)


def _display_data_quality(data: dict[str, pd.DataFrame | None]) -> None:
    st.title("🔍 Data Quality Monitoring")
    ticker_df = data.get("cnoi_tickers")
    stats = _compute_data_quality(ticker_df)
    if ticker_df is None or stats is None:
        st.warning("Ticker enrichment file missing.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coverage", f"{stats['coverage']:.1f}%")
    with col2:
        st.metric("Missing Tickers", stats["missing"])

    st.subheader("CNOI Score Distribution")
    if "CNOI" in ticker_df.columns:
        fig = px.histogram(ticker_df, x="CNOI", nbins=30, title="CNOI Scores")
        st.plotly_chart(fig, use_container_width=True)

    if stats["missing"]:
        st.subheader("Missing Ticker Records")
        st.dataframe(
            ticker_df.loc[ticker_df["ticker"].isna(), ["cik", "company_name"]],
            use_container_width=True,
        )


def main() -> None:
    """Entry point for Streamlit."""
    st.sidebar.title("📊 ACCT445 Showcase")
    st.sidebar.markdown("Bank Disclosure Opacity Trading System")
    st.sidebar.markdown("---")
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Decile Backtest", "Event Study", "Risk Metrics", "Data Quality"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    try:
        data = load_results()
    except Exception as exc:
        st.error(f"Error loading results: {exc}")
        st.info("Run `python src/runner/daily_backtest.py` to generate outputs.")
        st.stop()

    if page == "Overview":
        _display_overview(data)
    elif page == "Decile Backtest":
        _display_decile(data)
    elif page == "Event Study":
        _display_event_study(data)
    elif page == "Risk Metrics":
        _display_risk_metrics(data)
    else:
        _display_data_quality(data)


if __name__ == "__main__":
    main()
