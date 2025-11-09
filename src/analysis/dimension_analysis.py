"""CNOI dimension analysis utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas import PeriodDtype
from pandas.api.types import is_datetime64_any_dtype
from scipy import stats

from src.analysis.decile_backtest import run_decile_backtest
from src.utils.config import get_config_value
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

SIGNIFICANCE_THRESHOLD = 0.05
DEFAULT_WEIGHT_COL = "market_cap"

DIMENSIONS: dict[str, str] = {
    "D": "Discoverability (ease of finding CECL note)",
    "G": "Granularity (detail level)",
    "R": "Required Items (completeness)",
    "J": "Readability (complexity)",
    "T": "Table Density (use of tables vs text)",
    "S": "Stability (consistency over time)",
    "X": "Consistency (internal consistency)",
}


def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a datetime `date` column exists, deriving it from `quarter` when needed."""
    working = df.copy()

    if "date" in working.columns:
        date_values = working["date"]
        if isinstance(date_values.dtype, PeriodDtype):
            working["date"] = date_values.dt.to_timestamp()
        elif not is_datetime64_any_dtype(date_values):
            working["date"] = pd.to_datetime(date_values)
        return working

    if "quarter" in working.columns:
        quarter_values = working["quarter"]
        if isinstance(quarter_values.dtype, PeriodDtype):
            working["date"] = quarter_values.dt.to_timestamp()
        else:
            try:
                working["date"] = pd.PeriodIndex(quarter_values, freq="Q").to_timestamp()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Falling back to datetime conversion for quarter column: %s", exc)
                working["date"] = pd.to_datetime(quarter_values)
        return working

    raise DataValidationError("Dimension analysis requires either a 'date' or 'quarter' column.")


def _determine_weight_col(df: pd.DataFrame, weighting: str) -> str | None:
    """Return the column to use for weighting."""
    weighting = weighting.lower()
    if weighting not in {"equal", "value"}:
        raise ValueError("weighting must be either 'equal' or 'value'")

    if weighting == "value":
        if DEFAULT_WEIGHT_COL not in df.columns:
            raise DataValidationError(
                f"Value-weighted analysis requires '{DEFAULT_WEIGHT_COL}' in the dataframe."
            )
        return DEFAULT_WEIGHT_COL

    return None


def _compute_p_value(t_stat: float, n_obs: int | None) -> float:
    """Two-tailed p-value from a t-statistic."""
    if not np.isfinite(t_stat):
        return np.nan

    dof = max((n_obs or 0) - 1, 1)
    return 2 * (1 - stats.t.cdf(abs(t_stat), df=dof))


def analyze_single_dimension(
    df: pd.DataFrame,
    dimension: str,
    n_deciles: int | None = None,
    weighting: str | None = None,
) -> dict[str, Any]:
    """
    Run a decile backtest for a single CNOI dimension.

    Args:
        df: DataFrame with (ticker, date/quarter, ret_fwd, dimension scores)
        dimension: Dimension identifier (D, G, R, J, T, S, or X)
        n_deciles: Number of deciles to compute
        weighting: 'equal' or 'value' (value uses market_cap weights)

    Returns:
        Dictionary containing the decile summary and long/short statistics.
    """
    if dimension not in DIMENSIONS:
        raise ValueError(
            "Unknown dimension: " f"{dimension}. Must be one of {list(DIMENSIONS.keys())}."
        )

    n_deciles = n_deciles or int(get_config_value("backtest.n_deciles", 10))
    weighting = (weighting or str(get_config_value("backtest.weighting", "equal"))).lower()
    if weighting not in ("equal", "value"):
        raise DataValidationError("weighting must be either 'equal' or 'value'.")

    logger.info("Analyzing dimension %s - %s", dimension, DIMENSIONS[dimension])
    working = _ensure_date_column(df)

    weight_col = _determine_weight_col(working, weighting)

    required_cols = {"ticker", "ret_fwd", dimension}
    if weight_col:
        required_cols.add(weight_col)

    missing = required_cols.difference(working.columns)
    if missing:
        raise DataValidationError(
            "Dataset is missing required columns for dimension analysis: " f"{missing}"
        )

    cnoi_cols = ["ticker", "date", dimension]
    if weight_col:
        cnoi_cols.append(weight_col)

    cnoi_df = working[cnoi_cols].copy()
    returns_df = working[["ticker", "date", "ret_fwd"]].copy()

    summary, long_short_df = run_decile_backtest(
        cnoi_df,
        returns_df,
        score_col=dimension,
        return_col="ret_fwd",
        n_deciles=n_deciles,
        weight_col=weight_col,
    )

    if long_short_df.empty:
        raise DataValidationError("Long-short results are empty; check input data sufficiency.")

    long_short = long_short_df.iloc[0].to_dict()
    long_short["p_value"] = _compute_p_value(
        long_short.get("t_stat", np.nan),
        int(long_short.get("n_obs", 0)),
    )
    long_short["significant"] = (
        bool(long_short["p_value"] < SIGNIFICANCE_THRESHOLD)
        if np.isfinite(long_short["p_value"])
        else False
    )

    return {
        "dimension": dimension,
        "description": DIMENSIONS[dimension],
        "summary": summary,
        "long_short": long_short,
    }


def analyze_all_dimensions(
    df: pd.DataFrame,
    n_deciles: int | None = None,
    weighting: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the dimension analysis for every available dimension."""
    logger.info("Running dimension analysis across all %s dimensions", len(DIMENSIONS))

    results: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        if dimension not in df.columns:
            logger.warning("Dimension %s missing from dataframe; skipping.", dimension)
            continue
        try:
            results[dimension] = analyze_single_dimension(df, dimension, n_deciles, weighting)
        except Exception as exc:  # pragma: no cover - error paths depend on upstream data
            logger.error("Failed analyzing dimension %s: %s", dimension, exc)

    if not results:
        raise DataValidationError(
            "No dimensions were analyzed. Ensure the dataframe has dimension columns."
        )

    return results


def compare_dimensions(all_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison DataFrame ranked by absolute t-statistics."""
    if not all_results:
        raise ValueError("all_results is empty; run analyze_all_dimensions first.")

    records = []
    for dim, res in all_results.items():
        ls = res["long_short"]
        records.append(
            {
                "Dimension": dim,
                "Description": res["description"],
                "Long-Short Return": ls.get("mean_ret"),
                "T-Statistic": ls.get("t_stat"),
                "P-Value": ls.get("p_value"),
                "Significant (p<0.05)": bool(ls.get("significant")),
            }
        )

    comparison = pd.DataFrame(records)
    comparison["Ranking"] = (
        comparison["T-Statistic"].abs().rank(ascending=False, method="min").astype(int)
    )
    comparison = comparison.sort_values("Ranking").reset_index(drop=True)

    return comparison


def plot_dimension_comparison(
    comparison_df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Axes:
    """
    Plot horizontal bars of t-statistics for each dimension.

    Returns the matplotlib axes for further customization/testing.
    """
    if comparison_df.empty:
        raise ValueError("comparison_df is empty; cannot plot dimension comparison.")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [
        "#d62728" if p < SIGNIFICANCE_THRESHOLD else "#7f7f7f" for p in comparison_df["P-Value"]
    ]

    ax.barh(comparison_df["Dimension"], comparison_df["T-Statistic"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(-1.96, color="#d62728", linestyle="--", alpha=0.6, label="p = 0.05 threshold")
    ax.axvline(1.96, color="#d62728", linestyle="--", alpha=0.6)

    ax.set_xlabel("T-Statistic (Long-Short Spread)")
    ax.set_title("CNOI Dimension Analysis")
    ax.legend()

    plt.tight_layout()

    if save_path:
        output_path = Path(save_path)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info("Dimension comparison plot saved to %s", output_path)

    plt.show()
    return ax


def compute_dimension_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix of available dimension columns."""
    dimension_cols = [col for col in DIMENSIONS if col in df.columns]
    if not dimension_cols:
        raise ValueError("No dimension columns available to compute correlations.")

    corr = df[dimension_cols].corr()
    logger.info("Computed dimension correlation matrix for %s columns.", len(dimension_cols))
    return corr


def run_dimension_analysis_workflow(
    df: pd.DataFrame,
    n_deciles: int = 10,
    weighting: str = "equal",
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Utility wrapper to compute analysis + comparison in one call."""
    results = analyze_all_dimensions(df, n_deciles=n_deciles, weighting=weighting)
    comparison = compare_dimensions(results)
    return comparison, results


if __name__ == "__main__":  # pragma: no cover
    np.random.seed(42)
    tickers = [f"STOCK{i:03d}" for i in range(40)]
    quarters = pd.period_range("2020Q1", periods=16, freq="Q")

    rows = []
    for quarter in quarters:
        for ticker in tickers:
            dims = {dim: np.random.uniform(5, 15) for dim in DIMENSIONS}
            ret_fwd = -0.003 * dims["S"] - 0.0025 * dims["R"] + np.random.normal(0, 0.01)
            rows.append(
                {
                    "ticker": ticker,
                    "quarter": quarter,
                    "date": quarter.to_timestamp(),
                    **dims,
                    "ret_fwd": ret_fwd,
                    "market_cap": np.random.uniform(5e8, 5e10),
                }
            )

    demo_df = pd.DataFrame(rows)
    comparison_df, all_results = run_dimension_analysis_workflow(demo_df)

    logger.info("\n=== DIMENSION COMPARISON ===\n%s", comparison_df.to_string(index=False))
    logger.info("Top dimension: %s", comparison_df.iloc[0]["Dimension"])

    plot_dimension_comparison(comparison_df)
