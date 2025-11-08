"""
Event Study: SVB Collapse (March 2023)

Tests whether opaque banks (high CNOI) had worse cumulative abnormal returns
during the March 2023 banking crisis.

Event window: March 9-17, 2023 (SVB collapse + contagion week)

References:
- MacKinlay (1997): Event studies in economics and finance
- Campbell et al. (1997): The Econometrics of Financial Markets
"""

import numpy as np
import pandas as pd
from scipy import stats

from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger
from src.utils.validation import validate_event_inputs, validate_returns_schema

logger = get_logger(__name__)


def compute_market_model_params(
    returns_df: pd.DataFrame,
    market_returns: pd.Series,
    estimation_window_start: str,
    estimation_window_end: str,
    ticker_col: str = "ticker",
    return_col: str = "return",
) -> pd.DataFrame:
    """
    Estimate market model parameters (alpha, beta) for each stock.

    Model: R_it = α_i + β_i * R_mt + ε_it

    Args:
        returns_df: DataFrame with stock returns
        market_returns: Series with market returns (indexed by date)
        estimation_window_start: Start of estimation window (YYYY-MM-DD)
        estimation_window_end: End of estimation window (YYYY-MM-DD)
        ticker_col: Ticker column name
        return_col: Return column name

    Returns:
        DataFrame with columns: ticker, alpha, beta

    Example:
        >>> params = compute_market_model_params(
        ...     returns_df,
        ...     sp500_returns,
        ...     '2023-01-01',
        ...     '2023-03-08'
        ... )
    """
    validate_returns_schema(returns_df, required_columns=("date", ticker_col, return_col))

    if not isinstance(market_returns.index, pd.DatetimeIndex):
        raise DataValidationError("market_returns must have a DatetimeIndex")

    est_start = pd.Timestamp(estimation_window_start)
    est_end = pd.Timestamp(estimation_window_end)

    # Filter to estimation window
    est_df = returns_df[(returns_df["date"] >= est_start) & (returns_df["date"] <= est_end)].copy()

    # Merge with market returns
    market_df = market_returns.rename("market_return").to_frame().reset_index()
    market_df = market_df.rename(columns={"index": "date"})
    est_df = est_df.merge(market_df, on="date", how="inner")

    # Estimate OLS per ticker
    params = []
    for ticker in est_df[ticker_col].unique():
        ticker_data = est_df[est_df[ticker_col] == ticker]

        if len(ticker_data) < 20:  # Minimum observations
            continue

        # OLS: R_i = alpha + beta * R_m
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools.tools import add_constant

        X = add_constant(ticker_data["market_return"].values)
        y = ticker_data[return_col].values

        model = OLS(y, X).fit()

        params.append(
            {
                "ticker": ticker,
                "alpha": model.params[0],
                "beta": model.params[1],
                "r_squared": model.rsquared,
                "n_obs": len(ticker_data),
            }
        )

    if not params:
        raise DataValidationError("Insufficient data to estimate market model parameters.")

    logger.info("Estimated market model for %s tickers.", len(params))
    return pd.DataFrame(params)


def compute_abnormal_returns(
    returns_df: pd.DataFrame,
    market_returns: pd.Series,
    params_df: pd.DataFrame,
    event_start: str,
    event_end: str,
    ticker_col: str = "ticker",
    return_col: str = "return",
) -> pd.DataFrame:
    """
    Compute abnormal returns during event window.

    AR_it = R_it - (α_i + β_i * R_mt)

    Args:
        returns_df: DataFrame with stock returns
        market_returns: Series with market returns
        params_df: DataFrame from compute_market_model_params()
        event_start: Event window start (YYYY-MM-DD)
        event_end: Event window end (YYYY-MM-DD)

    Returns:
        DataFrame with columns: ticker, date, return, expected_return, abnormal_return

    Example:
        >>> ar_df = compute_abnormal_returns(
        ...     returns_df,
        ...     sp500_returns,
        ...     params,
        ...     '2023-03-09',
        ...     '2023-03-17'
        ... )
    """
    validate_returns_schema(returns_df, required_columns=("date", ticker_col, return_col))

    event_start_ts = pd.Timestamp(event_start)
    event_end_ts = pd.Timestamp(event_end)

    # Filter to event window
    event_df = returns_df[
        (returns_df["date"] >= event_start_ts) & (returns_df["date"] <= event_end_ts)
    ].copy()

    # Merge with market returns
    market_df = (
        market_returns.rename("market_return")
        .to_frame()
        .reset_index()
        .rename(columns={"index": "date"})
    )
    event_df = event_df.merge(market_df, on="date", how="inner")

    # Merge with params
    event_df = event_df.merge(params_df[["ticker", "alpha", "beta"]], on=ticker_col, how="inner")

    # Compute expected return
    event_df["expected_return"] = event_df["alpha"] + event_df["beta"] * event_df["market_return"]

    # Compute abnormal return
    event_df["abnormal_return"] = event_df[return_col] - event_df["expected_return"]

    logger.debug("Computed abnormal returns for %s observations.", len(event_df))
    return event_df[["ticker", "date", "return", "expected_return", "abnormal_return"]]


def compute_cumulative_abnormal_returns(
    ar_df: pd.DataFrame, ticker_col: str = "ticker"
) -> pd.DataFrame:
    """
    Compute cumulative abnormal returns (CAR) over event window.

    CAR_i = Σ AR_it

    Args:
        ar_df: DataFrame from compute_abnormal_returns()

    Returns:
        DataFrame with columns: ticker, CAR, n_days

    Example:
        >>> car_df = compute_cumulative_abnormal_returns(ar_df)
        >>> car_df.sort_values('CAR')
    """
    car = ar_df.groupby(ticker_col).agg({"abnormal_return": "sum", "date": "count"}).reset_index()

    car = car.rename(columns={"abnormal_return": "CAR", "date": "n_days"})

    logger.debug("Computed CAR for %s tickers.", len(car))
    return car


def test_cnoi_car_relationship(
    car_df: pd.DataFrame, cnoi_df: pd.DataFrame, pre_event_cutoff: str, n_quartiles: int = 4
) -> pd.DataFrame:
    """
    Test relationship between pre-event CNOI and CAR.

    Args:
        car_df: DataFrame from compute_cumulative_abnormal_returns()
        cnoi_df: DataFrame with CNOI scores
        pre_event_cutoff: Use CNOI scores before this date (YYYY-MM-DD)
        n_quartiles: Number of groups (4 = quartiles)

    Returns:
        DataFrame with CAR summary by CNOI quartile

    Example:
        >>> results = test_cnoi_car_relationship(
        ...     car_df,
        ...     cnoi_df,
        ...     pre_event_cutoff='2023-03-01'
        ... )
        >>> results.head()
    """
    # Get pre-event CNOI scores
    cutoff = pd.Timestamp(pre_event_cutoff)
    pre_cnoi = cnoi_df[cnoi_df["filing_date"] < cutoff].copy()

    # Latest CNOI per bank before event
    pre_cnoi = pre_cnoi.sort_values("filing_date").groupby("ticker").last().reset_index()

    # Merge with CAR
    merged = car_df.merge(pre_cnoi[["ticker", "CNOI"]], on="ticker", how="inner")

    # Assign quartiles
    merged["cnoi_quartile"] = pd.qcut(
        merged["CNOI"], q=n_quartiles, labels=[f"Q{i}" for i in range(1, n_quartiles + 1)]
    )

    # Summary by quartile
    summary = (
        merged.groupby("cnoi_quartile", observed=False)
        .agg({"CAR": ["mean", "std", "count"], "CNOI": ["mean", "min", "max"]})
        .reset_index()
    )

    summary.columns = ["_".join(col).strip("_") for col in summary.columns]

    # Test Q1 vs Q4
    q1_car = merged[merged["cnoi_quartile"] == "Q1"]["CAR"].values
    q4_car = merged[merged["cnoi_quartile"] == "Q4"]["CAR"].values

    if len(q1_car) > 1 and len(q4_car) > 1:
        t_stat, p_val = stats.ttest_ind(q1_car, q4_car)
        delta = (q4_car.mean() - q1_car.mean()) * 100
        logger.info(
            "Q1 vs Q4 CAR difference: %.2f%% (t=%.2f, p=%.4f)",
            delta,
            t_stat,
            p_val,
        )
    else:
        logger.warning("Not enough observations to run Q1 vs Q4 CAR test.")

    corr, corr_p = stats.pearsonr(merged["CNOI"], merged["CAR"])
    logger.info("CNOI vs CAR correlation %.3f (p=%.4f)", corr, corr_p)

    return summary


def run_event_study(
    returns_df: pd.DataFrame,
    market_returns: pd.Series,
    cnoi_df: pd.DataFrame,
    estimation_start: str = "2023-01-01",
    estimation_end: str = "2023-03-08",
    event_start: str = "2023-03-09",
    event_end: str = "2023-03-17",
    pre_event_cutoff: str = "2023-03-01",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Complete event study pipeline for SVB collapse.

    Args:
        returns_df: Stock returns DataFrame
        market_returns: Market index returns (e.g., S&P 500)
        cnoi_df: CNOI scores DataFrame
        estimation_start: Market model estimation start
        estimation_end: Market model estimation end (before event)
        event_start: Event window start (SVB collapse)
        event_end: Event window end
        pre_event_cutoff: Use CNOI scores before this date

    Returns:
        (car_by_quartile, individual_car) DataFrames

    Example:
        >>> summary, car_df = run_event_study(
        ...     returns_df,
        ...     sp500_returns,
        ...     cnoi_df
        ... )
    """
    validate_event_inputs(returns_df, market_returns)

    logger.info("=" * 60)
    logger.info("SVB Collapse Event Study")
    logger.info("=" * 60)
    logger.info("Estimation window: %s to %s", estimation_start, estimation_end)
    logger.info("Event window: %s to %s", event_start, event_end)

    # Step 1: Estimate market model
    logger.info("[1/4] Estimating market model parameters...")
    params = compute_market_model_params(
        returns_df, market_returns, estimation_start, estimation_end
    )
    logger.info("Estimated parameters for %s stocks", len(params))

    # Step 2: Compute abnormal returns
    logger.info("[2/4] Computing abnormal returns...")
    ar_df = compute_abnormal_returns(returns_df, market_returns, params, event_start, event_end)
    logger.info("Computed %s abnormal return observations", len(ar_df))

    # Step 3: Compute CAR
    logger.info("[3/4] Computing cumulative abnormal returns (CAR)...")
    car_df = compute_cumulative_abnormal_returns(ar_df)
    logger.info("CAR computed for %s stocks (mean %.2f%%)", len(car_df), car_df["CAR"].mean() * 100)

    # Step 4: Test CNOI relationship
    logger.info("[4/4] Testing CNOI vs CAR relationship...")
    quartile_summary = test_cnoi_car_relationship(car_df, cnoi_df, pre_event_cutoff)

    logger.info("=" * 60)
    logger.info("✓ Event study complete!")

    return quartile_summary, car_df


if __name__ == "__main__":  # pragma: no cover
    # Demo with simulated data
    logger.info("Event Study Demo (Simulated Data)")

    # Simulate market returns
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    market_ret = pd.Series(np.random.normal(0.0005, 0.015, len(dates)), index=dates)

    # SVB event: big negative market return
    svb_dates = pd.date_range("2023-03-09", "2023-03-17", freq="D")
    market_ret.loc[svb_dates] = np.random.normal(-0.02, 0.03, len(svb_dates))

    # Simulate stock returns
    tickers = [f"BANK{i:02d}" for i in range(20)]
    stock_data = []

    for ticker in tickers:
        alpha = np.random.uniform(-0.0005, 0.0005)
        beta = np.random.uniform(0.8, 1.5)

        for date in dates:
            ret = alpha + beta * market_ret.loc[date] + np.random.normal(0, 0.01)
            stock_data.append({"ticker": ticker, "date": date, "return": ret})

    returns_df = pd.DataFrame(stock_data)

    # Simulate CNOI scores
    cnoi_data = []
    for ticker in tickers:
        cnoi = np.random.uniform(8, 30)
        cnoi_data.append(
            {"ticker": ticker, "filing_date": pd.Timestamp("2023-02-15"), "CNOI": cnoi}
        )

    cnoi_df = pd.DataFrame(cnoi_data)

    # Run event study
    summary, car = run_event_study(returns_df, market_ret, cnoi_df)

    logger.info("Quartile Summary:\n%s", summary.to_string(index=False))
