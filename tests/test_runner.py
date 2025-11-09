"""Tests for automated runner + alerting utilities."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
from pytest import LogCaptureFixture

from src.runner.alerts import send_alert
from src.runner.daily_backtest import run_daily_update


def _build_sample_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    tickers = [f"T{idx:02d}" for idx in range(10)]
    filing_start = pd.Timestamp("2023-01-01")
    cnoi_rows = []
    for idx, ticker in enumerate(tickers):
        cnoi_rows.append(
            {
                "cik": 1000 + idx,
                "filing_date": filing_start + timedelta(days=idx * 3),
                "CNOI": 5.0 + idx,
                "ticker": ticker,
            }
        )
    cnoi_df = pd.DataFrame(cnoi_rows)

    return_rows = []
    base_dates = pd.date_range("2023-01-05", periods=4, freq="7D")
    for idx, ticker in enumerate(tickers):
        for step, date in enumerate(base_dates):
            return_rows.append(
                {
                    "date": date + timedelta(days=step),
                    "ticker": ticker,
                    "return": 0.01 - 0.0005 * step - 0.0002 * idx,
                }
            )
    returns_df = pd.DataFrame(return_rows).sort_values(["ticker", "date"]).reset_index(drop=True)
    returns_df["ret_fwd"] = returns_df.groupby("ticker")["return"].shift(-1)
    returns_df = returns_df.dropna(subset=["ret_fwd"]).reset_index(drop=True)
    return cnoi_df, returns_df


def test_send_alert_logs_warning(caplog: LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    send_alert("Test alert", "Test message")
    assert "ALERT: Test alert - Test message" in caplog.text


def test_run_daily_update_with_overrides(tmp_path: Path, caplog: LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    cnoi_df, returns_df = _build_sample_frames()

    stats = run_daily_update(
        cnoi_df=cnoi_df,
        returns_df=returns_df,
        results_dir=tmp_path,
    )

    assert stats is not None
    assert stats.get("portfolio") == "Long-Short (D1-D10)"
    assert (tmp_path / "decile_summary_latest.csv").exists()
    assert (tmp_path / "decile_long_short_latest.csv").exists()
