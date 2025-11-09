"""Panel regression utilities for evaluating CNOI signals."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from scipy import stats

from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def prepare_panel_data(
    df: pd.DataFrame,
    entity_col: str = "ticker",
    time_col: str = "quarter",
) -> pd.DataFrame:
    """
    Prepare data for panel regression with a sorted MultiIndex.

    Returns a copy of the dataframe indexed by (entity, time).
    """
    _validate_columns(df, [entity_col, time_col])

    panel_df = df.copy()

    time_values = panel_df[time_col]
    if isinstance(time_values.dtype, pd.PeriodDtype):
        panel_df[time_col] = time_values.dt.to_timestamp()
    elif not (is_numeric_dtype(time_values) or is_datetime64_any_dtype(time_values)):
        try:
            panel_df[time_col] = pd.to_datetime(time_values)
        except (TypeError, ValueError):
            panel_df[time_col] = time_values.astype("category").cat.codes

    panel_df = panel_df.set_index([entity_col, time_col]).sort_index()

    logger.info(
        "Panel data prepared with %s entities and %s periods",
        panel_df.index.get_level_values(0).nunique(),
        panel_df.index.get_level_values(1).nunique(),
    )
    return panel_df


def _drop_na(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cleaned = df[cols].dropna()
    if cleaned.empty:
        raise DataValidationError("Regression input is empty after dropping NaNs")
    return cleaned


def fixed_effects_regression(
    df: pd.DataFrame,
    dependent_var: str = "ret_fwd",
    independent_vars: list[str] | None = None,
    entity_effects: bool = True,
    time_effects: bool = True,
) -> dict[str, Any]:
    """Run Fixed-Effects regression using linearmodels.PanelOLS."""

    if independent_vars is None:
        independent_vars = ["CNOI"]

    required_cols = [dependent_var] + independent_vars
    _validate_columns(df, required_cols)

    if not isinstance(df.index, pd.MultiIndex):
        raise DataValidationError(
            "Fixed effects regression requires MultiIndex data; call prepare_panel_data first."
        )

    clean = _drop_na(df, required_cols)
    y = clean[dependent_var]
    X = clean[independent_vars]

    logger.info(
        "Running fixed effects regression (%s obs, %s vars)",
        len(clean),
        len(independent_vars),
    )

    model = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects)
    results = model.fit(
        cov_type="clustered",
        cluster_entity=entity_effects,
        cluster_time=time_effects,
    )

    if entity_effects and time_effects:
        effects = "twoway"
    elif entity_effects:
        effects = "entity"
    elif time_effects:
        effects = "time"
    else:
        effects = "none"

    return {
        "coefficients": results.params.to_dict(),
        "std_errors": results.std_errors.to_dict(),
        "t_stats": results.tstats.to_dict(),
        "p_values": results.pvalues.to_dict(),
        "r_squared": results.rsquared,
        "n_obs": results.nobs,
        "effects": effects,
        "model": results,
    }


def fama_macbeth_regression(
    df: pd.DataFrame,
    dependent_var: str = "ret_fwd",
    independent_vars: list[str] | None = None,
    time_col: str = "quarter",
    min_obs: int | None = None,
) -> dict[str, Any]:
    """Run Fama-MacBeth two-step cross-sectional regression."""

    if independent_vars is None:
        independent_vars = ["CNOI"]

    required_cols = [dependent_var, time_col] + independent_vars
    _validate_columns(df, required_cols)

    if min_obs is None:
        min_obs = len(independent_vars) + 2

    df_sorted = df.sort_values(time_col)
    periods = df_sorted[time_col].unique()
    period_coeffs: dict[str, list[float]] = defaultdict(list)

    for period in periods:
        period_data = df_sorted[df_sorted[time_col] == period]
        period_data = period_data.dropna(subset=required_cols)

        if len(period_data) < min_obs:
            logger.warning("Skipping %s: only %s observations", period, len(period_data))
            continue

        y = period_data[dependent_var]
        X = sm.add_constant(period_data[independent_vars], prepend=True)

        model = sm.OLS(y, X).fit()
        params = model.params.to_dict()

        const_val = params.get("const", np.nan)
        if np.isnan(const_val):
            logger.warning("No constant estimated for %s; appending NaN", period)
        period_coeffs["const"].append(const_val)

        for var in independent_vars:
            value = params.get(var, np.nan)
            if np.isnan(value):
                logger.warning("Coefficient for %s missing in %s; appending NaN", var, period)
            period_coeffs[var].append(value)

    n_periods = len(next(iter(period_coeffs.values()), []))
    if n_periods == 0:
        raise DataValidationError("No valid periods for Fama-MacBeth regression")

    summary: dict[str, dict[str, float]] = {}
    for var, values in period_coeffs.items():
        series = pd.Series(values, dtype="float64").dropna()
        if series.empty:
            logger.warning("Skipping %s: no valid coefficients", var)
            continue

        mean_coef = series.mean()
        n = len(series)
        if n < 2:
            se = np.nan
            t_stat = np.nan
            p_value = np.nan
        else:
            se = series.std(ddof=1) / np.sqrt(n)
            if se == 0 or np.isnan(se):
                t_stat = np.inf
                p_value = 0.0
            else:
                t_stat = mean_coef / se
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))

        summary[var] = {
            "coefficient": mean_coef,
            "std_error": se,
            "t_stat": t_stat,
            "p_value": p_value,
        }

    logger.info(
        "Fama-MacBeth regression complete across %s periods",
        len(next(iter(period_coeffs.values()))),
    )

    return {
        "coefficients": {k: v["coefficient"] for k, v in summary.items()},
        "std_errors": {k: v["std_error"] for k, v in summary.items()},
        "t_stats": {k: v["t_stat"] for k, v in summary.items()},
        "p_values": {k: v["p_value"] for k, v in summary.items()},
        "n_periods": len(next(iter(period_coeffs.values()))),
        "period_coefficients": period_coeffs,
    }


def driscoll_kraay_regression(
    df: pd.DataFrame,
    dependent_var: str = "ret_fwd",
    independent_vars: list[str] | None = None,
    max_lags: int = 4,
) -> dict[str, Any]:
    """Run regression with Driscoll-Kraay standard errors."""

    if independent_vars is None:
        independent_vars = ["CNOI"]

    required_cols = [dependent_var] + independent_vars
    _validate_columns(df, required_cols)

    if not isinstance(df.index, pd.MultiIndex):
        raise DataValidationError("Driscoll-Kraay regression expects MultiIndex panel data.")

    clean = _drop_na(df, required_cols)
    y = clean[dependent_var]
    X = clean[independent_vars]

    logger.info("Running Driscoll-Kraay regression (%s obs, max_lags=%s)", len(clean), max_lags)
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=max_lags)

    return {
        "coefficients": results.params.to_dict(),
        "std_errors": results.std_errors.to_dict(),
        "t_stats": results.tstats.to_dict(),
        "p_values": results.pvalues.to_dict(),
        "r_squared": results.rsquared,
        "n_obs": results.nobs,
        "max_lags": max_lags,
        "model": results,
    }


def run_all_panel_regressions(
    df: pd.DataFrame,
    dependent_var: str = "ret_fwd",
    independent_vars: list[str] | None = None,
    entity_col: str = "ticker",
    time_col: str = "quarter",
) -> dict[str, Any]:
    """Run FE, Fama-MacBeth, and Driscoll-Kraay regressions for comparison."""

    if independent_vars is None:
        independent_vars = ["CNOI"]

    logger.info("Executing panel regression suite on %s observations", len(df))
    panel_df = prepare_panel_data(df, entity_col=entity_col, time_col=time_col)

    fe_results = fixed_effects_regression(
        panel_df,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
    )
    fm_results = fama_macbeth_regression(
        df,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        time_col=time_col,
    )
    dk_results = driscoll_kraay_regression(
        panel_df,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
    )

    comparison = pd.DataFrame(
        {
            "FE_coef": fe_results["coefficients"],
            "FE_tstat": fe_results["t_stats"],
            "FM_coef": fm_results["coefficients"],
            "FM_tstat": fm_results["t_stats"],
            "DK_coef": dk_results["coefficients"],
            "DK_tstat": dk_results["t_stats"],
        }
    )

    logger.info("Panel regression comparison:\n%s", comparison)

    return {
        "FE": fe_results,
        "FM": fm_results,
        "DK": dk_results,
        "comparison": comparison,
    }


def _demo() -> None:  # pragma: no cover - smoke test
    np.random.seed(42)
    n_entities = 30
    n_periods = 12
    tickers = [f"BANK{i:02d}" for i in range(n_entities)]
    quarters = pd.period_range("2020Q1", periods=n_periods, freq="Q")

    rows = []
    for ticker in tickers:
        entity_alpha = np.random.normal(0, 0.01)
        for quarter in quarters:
            time_effect = np.random.normal(0, 0.005)
            cnoi = np.random.uniform(5, 25)
            log_mcap = np.random.uniform(20, 25)
            ret = (
                entity_alpha
                + time_effect
                - 0.002 * cnoi
                + 0.0005 * log_mcap
                + np.random.normal(0, 0.01)
            )
            rows.append(
                {
                    "ticker": ticker,
                    "quarter": quarter,
                    "CNOI": cnoi,
                    "log_mcap": log_mcap,
                    "ret_fwd": ret,
                }
            )

    df = pd.DataFrame(rows)
    results = run_all_panel_regressions(df, independent_vars=["CNOI", "log_mcap"])
    logger.info("Demo comparison:\n%s", results["comparison"])


if __name__ == "__main__":  # pragma: no cover
    _demo()
