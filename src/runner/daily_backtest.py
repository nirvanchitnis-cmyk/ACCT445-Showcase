"""Automated daily backtest runner with scheduling + alerting."""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import schedule

from src.analysis.decile_backtest import run_decile_backtest
from src.data.cik_ticker_mapper import enrich_cnoi_with_tickers
from src.runner.alerts import send_alert
from src.utils.config import get_config_value
from src.utils.data_loader import (
    compute_forward_returns,
    load_cnoi_data,
    merge_cnoi_with_returns,
)
from src.utils.logger import get_logger
from src.utils.market_data import fetch_bulk_data, validate_data_quality

LOG_FILE = Path("logs/runner.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
LOGGER = get_logger(__name__, log_file=LOG_FILE)


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        LOGGER.warning("Unable to parse date '%s' from config; falling back to defaults.", value)
        return None


def _determine_window() -> tuple[str, str]:
    now = datetime.utcnow()
    end_dt = _parse_date(get_config_value("market_data.end_date")) or now
    lookback_days = int(get_config_value("runner.lookback_days", 365))
    start_override = _parse_date(get_config_value("market_data.start_date"))
    start_dt = start_override or (end_dt - timedelta(days=lookback_days))
    if start_dt >= end_dt:
        start_dt = end_dt - timedelta(days=max(lookback_days, 30))
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _prepare_cnoi_data(cnoi_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if cnoi_df is None:
        cnoi_path = Path(get_config_value("data.cnoi_file", "config/sample_cnoi.csv"))
        LOGGER.info("Loading CNOI data from %s", cnoi_path)
        df = load_cnoi_data(str(cnoi_path))
    else:
        df = cnoi_df.copy()
        if "filing_date" in df.columns:
            df["filing_date"] = pd.to_datetime(df["filing_date"])

    if "ticker" not in df.columns or df["ticker"].isna().any():
        LOGGER.info("Ticker column missing/incomplete; enriching via SEC mapping.")
        df = enrich_cnoi_with_tickers(df)

    df = df.dropna(subset=["ticker", "filing_date"]).copy()
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _prepare_returns_df(returns_df: pd.DataFrame) -> pd.DataFrame:
    df = returns_df.copy()
    if "ret" in df.columns and "return" not in df.columns:
        df = df.rename(columns={"ret": "return"})
    if "close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"close": "price"})
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if "volume" not in df.columns:
        df["volume"] = pd.NA

    if "ret_fwd" not in df.columns:
        df = compute_forward_returns(df, horizon=1, frequency="daily")

    df = df.dropna(subset=["ret_fwd"]).reset_index(drop=True)
    return df


def _fetch_and_prepare_returns(tickers: list[str]) -> pd.DataFrame:
    start_date, end_date = _determine_window()
    LOGGER.info(
        "Fetching market data for %s tickers (%s → %s)",
        len(tickers),
        start_date,
        end_date,
    )
    market_df = fetch_bulk_data(
        tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=bool(get_config_value("market_data.use_cache", True)),
        max_retries=int(get_config_value("market_data.max_retries", 3)),
        parallel=bool(get_config_value("runner.enable_parallel_fetch", True)),
        n_jobs=-1,
    )
    validate_data_quality(market_df)
    return _prepare_returns_df(market_df)


def _save_results(summary: pd.DataFrame, long_short: pd.DataFrame, *, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "decile_summary_latest.csv"
    long_short_path = results_dir / "decile_long_short_latest.csv"
    summary.to_csv(summary_path, index=False)
    long_short.to_csv(long_short_path, index=False)
    LOGGER.info(
        "Saved latest decile outputs: %s rows summary, %s rows long-short.",
        len(summary),
        len(long_short),
    )


def run_daily_update(
    *,
    cnoi_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
    results_dir: Path | None = None,
) -> dict[str, Any] | None:
    """Execute the daily refresh pipeline and return long-short stats."""
    LOGGER.info("%s", "=" * 50)
    LOGGER.info("Daily update started: %s", datetime.now())
    LOGGER.info("%s", "=" * 50)

    try:
        cnoi_prepped = _prepare_cnoi_data(cnoi_df)
        tickers = sorted(set(cnoi_prepped["ticker"].dropna()))
        if not tickers:
            raise ValueError("No tickers available after preparing CNOI data.")

        if returns_df is None:
            returns_prepped = _fetch_and_prepare_returns(tickers)
        else:
            LOGGER.info(
                "Using provided returns dataframe with %s rows for runner execution.",
                len(returns_df),
            )
            returns_prepped = _prepare_returns_df(returns_df)

        lag_days = int(get_config_value("backtest.lag_days", 2))
        merged = merge_cnoi_with_returns(cnoi_prepped, returns_prepped, lag_days=lag_days)
        if "date_y" in merged.columns:
            merged = merged.rename(columns={"date_y": "date"})
        if "date_x" in merged.columns:
            if "date" in merged.columns:
                merged = merged.drop(columns=["date_x"])
            else:
                merged = merged.rename(columns={"date_x": "date"})
        merged = merged.dropna(subset=["ret_fwd"])
        if merged.empty:
            send_alert("Runner warning", "No merged observations for daily update.")
            return None

        cnoi_bt = merged[["ticker", "date", "CNOI"]].copy()
        returns_bt = merged[["ticker", "date", "ret_fwd"]].copy()
        returns_bt.rename(columns={"ret_fwd": "ret_fwd"}, inplace=True)

        n_deciles = int(get_config_value("backtest.n_deciles", 10))
        summary, long_short_df = run_decile_backtest(
            cnoi_bt,
            returns_bt,
            score_col="CNOI",
            return_col="ret_fwd",
            n_deciles=n_deciles,
        )

        if long_short_df.empty:
            send_alert("Runner warning", "Long-short results empty; insufficient data.")
            return None

        long_short_stats = long_short_df.iloc[0].to_dict()
        results_path = results_dir or Path(get_config_value("data.results_dir", "results"))
        _save_results(summary, long_short_df, results_dir=results_path)

        threshold = float(get_config_value("runner.anomaly_tstat_threshold", 1.0))
        t_stat = long_short_stats.get("t_stat")
        if pd.notna(t_stat) and abs(float(t_stat)) < threshold:
            send_alert(
                "CNOI signal weakened",
                f"t-stat dropped to {float(t_stat):.2f}",
            )

        LOGGER.info("Daily update completed successfully")
        return long_short_stats

    except Exception as exc:  # noqa: BLE001 - top-level guard for scheduler
        LOGGER.error("Daily update failed: %s", exc, exc_info=True)
        send_alert("Daily backtest failed", str(exc))
        return None


def schedule_daily_updates(run_on_start: bool = True) -> None:
    """Schedule daily updates at configured time (default 6 PM ET)."""
    schedule_time = str(get_config_value("runner.schedule_time", "18:00"))
    if run_on_start:
        run_daily_update()

    schedule.every().day.at(schedule_time).do(run_daily_update)
    LOGGER.info("Scheduler started. Updates scheduled daily at %s", schedule_time)

    try:
        while True:
            schedule.run_pending()
            time.sleep(60)
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.info("Scheduler interrupted; shutting down runner.")


if __name__ == "__main__":  # pragma: no cover
    schedule_daily_updates()
