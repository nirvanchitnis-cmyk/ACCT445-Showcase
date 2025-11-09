"""Robustness checks for portfolio backtests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import tqdm

from src.analysis.decile_backtest import run_decile_backtest
from src.utils.caching import disk_cache, hash_dataframe
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_CACHE_DIR = PROJECT_ROOT / ".cache" / "robustness"


def _prepare_backtest_inputs(
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    weight_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract the CNOI and returns frames expected by run_decile_backtest."""

    required = {"ticker", "date", score_col, return_col}
    if weight_col:
        required.add(weight_col)

    missing = required.difference(df.columns)
    if missing:
        raise DataValidationError(f"Robustness dataset missing required columns: {sorted(missing)}")

    cnoi_cols = ["ticker", "date", score_col]
    if weight_col:
        cnoi_cols.append(weight_col)

    cnoi_df = df[cnoi_cols].copy()
    returns_df = df[["ticker", "date", return_col]].copy()
    return cnoi_df, returns_df


def _run_backtest_once(
    df: pd.DataFrame,
    score_col: str,
    *,
    return_col: str,
    n_deciles: int,
    weight_col: str | None,
    lags: int,
) -> pd.DataFrame:
    """Helper that executes the decile backtest and returns the long-short row."""

    cnoi_df, returns_df = _prepare_backtest_inputs(df, score_col, return_col, weight_col)
    _, long_short = run_decile_backtest(
        cnoi_df,
        returns_df,
        score_col=score_col,
        return_col=return_col,
        n_deciles=n_deciles,
        weight_col=weight_col,
        lags=lags,
    )

    if long_short.empty:
        raise DataValidationError("Long-short results are empty.")

    return long_short.iloc[0]


def _build_cache_key(
    method: str,
    df: pd.DataFrame,
    score_col: str,
    return_col: str,
    payload: dict[str, Any],
) -> str:
    """Create a deterministic cache key for robustness routines."""

    cols = ["ticker", "date", score_col, return_col]
    weight_col = payload.get("weight_col")
    if weight_col:
        cols.append(weight_col)

    try:
        subset = df[cols].copy()
    except KeyError:
        subset = df.copy()
    content_hash = hash_dataframe(subset)
    data = {
        "method": method,
        "df": content_hash,
        "score_col": score_col,
        "return_col": return_col,
    }
    data.update(payload)
    return hashlib.md5(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def _bootstrap_cache_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    df, score_col = args[:2]
    return_col = kwargs.get("return_col", "ret_fwd")
    payload = {
        "n_deciles": kwargs.get("n_deciles", 10),
        "n_bootstrap": kwargs.get("n_bootstrap", 1000),
        "confidence": kwargs.get("confidence", 0.95),
        "random_seed": kwargs.get("random_seed", 42),
        "weight_col": kwargs.get("weight_col"),
        "lags": kwargs.get("lags", 3),
    }
    return _build_cache_key("bootstrap", df, score_col, return_col, payload)


def _permutation_cache_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    df, score_col = args[:2]
    return_col = kwargs.get("return_col", "ret_fwd")
    payload = {
        "n_deciles": kwargs.get("n_deciles", 10),
        "n_permutations": kwargs.get("n_permutations", 1000),
        "random_seed": kwargs.get("random_seed", 42),
        "weight_col": kwargs.get("weight_col"),
        "lags": kwargs.get("lags", 3),
    }
    return _build_cache_key("permutation", df, score_col, return_col, payload)


@disk_cache(
    ROBUSTNESS_CACHE_DIR,
    key_func=_bootstrap_cache_key,
    disable_cache_kwarg="use_cache",
)
def bootstrap_backtest(
    df: pd.DataFrame,
    score_col: str,
    *,
    return_col: str = "ret_fwd",
    n_deciles: int = 10,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int | None = 42,
    weight_col: str | None = None,
    lags: int = 3,
    progress: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Bootstrap resampling for long-short spread confidence intervals.

    Args:
        use_cache: When False, bypass the on-disk cache and recompute the bootstrap.

    Returns:
        Dictionary containing mean, ci bounds, and the bootstrap distribution.

    Notes:
        Set ``use_cache=False`` to force recomputation instead of reading a cached result.
    """

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    if df.empty:
        raise DataValidationError("Input dataframe is empty.")

    rng = np.random.default_rng(random_seed)
    spreads: list[float] = []

    iterator = range(n_bootstrap)
    if progress:
        iterator = tqdm(iterator, desc="Bootstrap", leave=False)

    for _ in iterator:
        sample_idx = rng.integers(0, len(df), len(df))
        sample = df.iloc[sample_idx].reset_index(drop=True)
        try:
            row = _run_backtest_once(
                sample,
                score_col,
                return_col=return_col,
                n_deciles=n_deciles,
                weight_col=weight_col,
                lags=lags,
            )
            spreads.append(float(row["mean_ret"]))
        except DataValidationError as exc:
            logger.debug("Skipping bootstrap sample: %s", exc)
            continue

    if not spreads:
        raise DataValidationError("Bootstrap produced no valid samples.")

    distribution = np.asarray(spreads)
    alpha = 1 - confidence
    ci_lower, ci_upper = np.quantile(distribution, [alpha / 2, 1 - alpha / 2])

    logger.info(
        "Bootstrap complete (%s draws). Mean=%.4f, CI=[%.4f, %.4f]",
        len(distribution),
        distribution.mean(),
        ci_lower,
        ci_upper,
    )

    return {
        "mean": float(distribution.mean()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "distribution": distribution,
        "n_draws": len(distribution),
    }


@disk_cache(
    ROBUSTNESS_CACHE_DIR,
    key_func=_permutation_cache_key,
    disable_cache_kwarg="use_cache",
)
def permutation_test(
    df: pd.DataFrame,
    score_col: str,
    *,
    return_col: str = "ret_fwd",
    n_deciles: int = 10,
    n_permutations: int = 1000,
    random_seed: int | None = 42,
    weight_col: str | None = None,
    lags: int = 3,
    progress: bool = True,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Permutation test where score labels are shuffled to form the null distribution.

    Args:
        use_cache: When False, bypass the cache and perform a fresh permutation run.

    Notes:
        Set ``use_cache=False`` to force recomputation instead of using cached values.
    """

    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive.")

    observed_row = _run_backtest_once(
        df,
        score_col,
        return_col=return_col,
        n_deciles=n_deciles,
        weight_col=weight_col,
        lags=lags,
    )
    observed = float(observed_row["mean_ret"])
    logger.info("Observed long-short spread: %.4f", observed)

    rng = np.random.default_rng(random_seed)
    null_spreads: list[float] = []

    iterator = range(n_permutations)
    if progress:
        iterator = tqdm(iterator, desc="Permutation", leave=False)

    for _ in iterator:
        shuffled = df.copy()
        shuffled[score_col] = rng.permutation(shuffled[score_col].values)
        try:
            row = _run_backtest_once(
                shuffled,
                score_col,
                return_col=return_col,
                n_deciles=n_deciles,
                weight_col=weight_col,
                lags=lags,
            )
            null_spreads.append(float(row["mean_ret"]))
        except DataValidationError as exc:
            logger.debug("Skipping permutation sample: %s", exc)
            continue

    if not null_spreads:
        raise DataValidationError("Permutation test produced no valid samples.")

    null_distribution = np.asarray(null_spreads)
    p_value = float((np.abs(null_distribution) >= abs(observed)).mean())
    logger.info("Permutation test complete: p-value=%.4f", p_value)

    return {
        "observed": observed,
        "p_value": p_value,
        "null_distribution": null_distribution,
        "n_draws": len(null_distribution),
    }


def subsample_analysis(
    df: pd.DataFrame,
    score_col: str,
    *,
    split_col: str,
    split_values: Sequence[Any] | None = None,
    return_col: str = "ret_fwd",
    n_deciles: int = 10,
    min_obs: int = 60,
    weight_col: str | None = None,
    lags: int = 3,
) -> pd.DataFrame:
    """
    Evaluate long-short performance across different subsamples (e.g., regimes).
    """

    if split_col not in df.columns:
        raise DataValidationError(f"{split_col!r} column not present in dataframe.")

    if split_values is None:
        split_values = df[split_col].dropna().unique().tolist()

    results = []
    for value in split_values:
        subset = df[df[split_col] == value]
        if len(subset) < min_obs:
            logger.warning(
                "Skipping %s=%s due to insufficient observations (%s).",
                split_col,
                value,
                len(subset),
            )
            continue

        try:
            row = _run_backtest_once(
                subset,
                score_col,
                return_col=return_col,
                n_deciles=n_deciles,
                weight_col=weight_col,
                lags=lags,
            )
            n_obs = int(row.get("n_obs", len(subset)))
            results.append(
                {
                    split_col: value,
                    "n_obs": n_obs,
                    "long_short_ret": float(row["mean_ret"]),
                    "t_stat": float(row["t_stat"]),
                    "p_value": _two_tailed_p_value(row.get("t_stat", np.nan), n_obs),
                }
            )
        except DataValidationError as exc:
            logger.warning("Skipping %s=%s due to error: %s", split_col, value, exc)
            continue

    if not results:
        raise DataValidationError("No subsample results were generated.")

    return pd.DataFrame(results)


def monte_carlo_long_short(
    returns: pd.Series,
    *,
    horizon: int = 12,
    n_scenarios: int = 1000,
    random_seed: int | None = 42,
) -> dict[str, Any]:
    """
    Monte Carlo resampling of long-short returns to simulate future paths.
    """

    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if n_scenarios <= 0:
        raise ValueError("n_scenarios must be positive.")

    series = returns.dropna()
    if series.empty:
        raise DataValidationError("Returns series is empty.")

    rng = np.random.default_rng(random_seed)
    draws = rng.choice(series.values, size=(n_scenarios, horizon), replace=True)
    scenario_returns = draws.sum(axis=1)

    return {
        "mean": float(scenario_returns.mean()),
        "p05": float(np.percentile(scenario_returns, 5)),
        "p95": float(np.percentile(scenario_returns, 95)),
        "distribution": scenario_returns,
        "n_draws": n_scenarios,
    }


def _two_tailed_p_value(t_stat: float, n_obs: int) -> float:
    """Compute a two-tailed p-value given a t-statistic."""
    if not np.isfinite(t_stat):
        return np.nan
    df = max(n_obs - 1, 1)
    return float(2 * (1 - stats.t.cdf(abs(t_stat), df=df)))


__all__ = [
    "bootstrap_backtest",
    "permutation_test",
    "subsample_analysis",
    "monte_carlo_long_short",
]


if __name__ == "__main__":  # pragma: no cover
    np.random.seed(7)
    tickers = [f"BK{i:02d}" for i in range(25)]
    dates = pd.period_range("2021Q1", periods=12, freq="Q")

    records = []
    for date in dates:
        for ticker in tickers:
            score = np.random.uniform(10, 35)
            noise = np.random.normal(0, 0.02)
            ret = -0.002 * (score - 20) + noise
            records.append(
                {
                    "ticker": ticker,
                    "date": date.to_timestamp(),
                    "CNOI": score,
                    "ret_fwd": ret,
                    "regime": "stress" if date.year == 2022 else "calm",
                }
            )

    demo_df = pd.DataFrame(records)

    boot = bootstrap_backtest(demo_df, "CNOI", n_bootstrap=200, progress=False)
    logger.info("Bootstrap CI: [%.4f, %.4f]", boot["ci_lower"], boot["ci_upper"])

    perm = permutation_test(demo_df, "CNOI", n_permutations=200, progress=False)
    logger.info("Permutation p-value: %.4f", perm["p_value"])

    subsample = subsample_analysis(
        demo_df,
        "CNOI",
        split_col="regime",
        split_values=["calm", "stress"],
    )
    logger.info("Subsample analysis:\n%s", subsample.to_string(index=False))

    ls_row = _run_backtest_once(
        demo_df,
        "CNOI",
        return_col="ret_fwd",
        n_deciles=10,
        weight_col=None,
        lags=3,
    )
    ls_returns = pd.Series(
        np.random.normal(loc=ls_row["mean_ret"], scale=0.01, size=24),
        name="ls_return",
    )
    mc = monte_carlo_long_short(ls_returns, horizon=4, n_scenarios=500)
    logger.info("Monte Carlo p05/p95: %.4f / %.4f", mc["p05"], mc["p95"])
