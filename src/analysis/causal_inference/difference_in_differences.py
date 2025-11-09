"""
Difference-in-differences estimation with two-way clustered SEs.

References:
- Petersen (2009) - Estimating standard errors in finance panels
- Cameron & Miller (2015) - Practitioners' guide to cluster-robust inference
- Bertrand, Duflo, Mullainathan (2004) - How much should we trust DiD estimates?
"""

from typing import Any

import numpy as np
import pandas as pd
from linearmodels import PanelOLS

from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_did_data(
    panel_df: pd.DataFrame,
    treatment_col: str = "treated",
    post_col: str = "post_cecl",
    outcome_col: str = "ret_next_quarter",
    controls: list[str] | None = None,
) -> pd.DataFrame:
    """
    Prepare panel data for DiD regression.

    Creates interaction term: treated × post

    Args:
        panel_df: Panel data with bank-quarter observations
        treatment_col: Treatment indicator (1 = CECL adopter)
        post_col: Post-treatment indicator (1 = after adoption)
        outcome_col: Dependent variable (returns, volatility, etc.)
        controls: List of control variables

    Returns:
        DataFrame with 'treat_post' interaction column

    Example:
        >>> df = prepare_did_data(
        ...     panel_df,
        ...     treatment_col='cecl_adopter',
        ...     post_col='post_2020q1',
        ...     outcome_col='quarterly_return'
        ... )
        >>> df['treat_post'].sum()  # Count treated × post observations
    """
    if treatment_col not in panel_df.columns:
        raise ValueError(f"Treatment column '{treatment_col}' not found in panel_df")
    if post_col not in panel_df.columns:
        raise ValueError(f"Post column '{post_col}' not found in panel_df")
    if outcome_col not in panel_df.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in panel_df")

    df = panel_df.copy()

    # Create DiD interaction term
    df["treat_post"] = df[treatment_col] * df[post_col]

    # Ensure numeric types
    df[treatment_col] = df[treatment_col].astype(float)
    df[post_col] = df[post_col].astype(float)
    df["treat_post"] = df["treat_post"].astype(float)

    logger.info(
        "DiD data prepared: %d observations, %d treated, %d post-treatment, %d treated×post",
        len(df),
        df[treatment_col].sum(),
        df[post_col].sum(),
        df["treat_post"].sum(),
    )

    return df


def run_did_regression(
    did_df: pd.DataFrame,
    outcome: str,
    entity_col: str = "cik",
    time_col: str = "quarter",
    controls: list[str] | None = None,
    cluster_entity: bool = True,
    cluster_time: bool = True,
    entity_fe: bool = True,
    time_fe: bool = True,
) -> Any:
    """
    Estimate DiD with two-way fixed effects and clustered SEs.

    Model:
        Y_it = α + β1·Treat_i + β2·Post_t + δ·(Treat × Post)_it + γ·X_it + μ_i + λ_t + ε_it

    Where:
        - δ is the DiD estimator (treatment effect)
        - μ_i = entity fixed effects (bank FE)
        - λ_t = time fixed effects (quarter FE)
        - SEs clustered by bank AND time (two-way)

    Args:
        did_df: Panel data from prepare_did_data()
        outcome: Dependent variable name
        entity_col: Entity identifier (e.g., 'cik')
        time_col: Time identifier (e.g., 'quarter')
        controls: Control variables (e.g., log_mcap, leverage, ROA)
        cluster_entity: Cluster standard errors by entity
        cluster_time: Cluster standard errors by time
        entity_fe: Include entity fixed effects
        time_fe: Include time fixed effects

    Returns:
        PanelOLSResults with clustered SEs

    Example:
        >>> result = run_did_regression(
        ...     did_df,
        ...     outcome='quarterly_return',
        ...     controls=['log_mcap', 'leverage']
        ... )
        >>> print(result.summary)
    """
    # Validate required columns
    required_cols = [outcome, entity_col, time_col, "treat_post"]
    missing_cols = [col for col in required_cols if col not in did_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for controls
    if controls:
        missing_controls = [c for c in controls if c not in did_df.columns]
        if missing_controls:
            raise ValueError(f"Missing control variables: {missing_controls}")

    # Prepare data for PanelOLS
    df = did_df.copy()

    # Convert time column to datetime if it's a string
    if df[time_col].dtype == "object":
        df[time_col] = pd.to_datetime(df[time_col])

    # Set multi-index
    df = df.set_index([entity_col, time_col])

    # Build formula
    formula = f"{outcome} ~ treat_post"

    # Add controls
    if controls:
        formula += " + " + " + ".join(controls)

    # Add fixed effects
    if entity_fe:
        formula += " + EntityEffects"
    if time_fe:
        formula += " + TimeEffects"

    logger.info("DiD formula: %s", formula)
    logger.info(
        "Clustering: entity=%s, time=%s",
        cluster_entity,
        cluster_time,
    )

    # Estimate with PanelOLS
    try:
        model = PanelOLS.from_formula(formula, data=df)

        # Determine clustering type
        if cluster_entity and cluster_time:
            cov_type = "clustered"
            result = model.fit(cov_type=cov_type, cluster_entity=True, cluster_time=True)
        elif cluster_entity:
            cov_type = "clustered"
            result = model.fit(cov_type=cov_type, cluster_entity=True)
        elif cluster_time:
            cov_type = "clustered"
            result = model.fit(cov_type=cov_type, cluster_time=True)
        else:
            result = model.fit(cov_type="robust")

        logger.info("DiD estimation complete. R-squared: %.4f", result.rsquared)

        return result

    except Exception as e:
        logger.error("DiD regression failed: %s", str(e))
        raise


def did_summary_table(result: Any) -> pd.DataFrame:
    """
    Extract key coefficients from DiD regression.

    Args:
        result: PanelOLSResults from run_did_regression()

    Returns:
        DataFrame with columns: ['coef', 'se', 't_stat', 'p_value', '95%_CI_lower', '95%_CI_upper']
        Index: ['treat_post (DiD)', 'controls...']

    Example:
        >>> summary = did_summary_table(result)
        >>> print(summary.loc['treat_post'])
    """
    # Extract confidence intervals
    ci = result.conf_int()

    summary = pd.DataFrame(
        {
            "coef": result.params,
            "se": result.std_errors,
            "t_stat": result.tstats,
            "p_value": result.pvalues,
            "95%_CI_lower": ci.iloc[:, 0],
            "95%_CI_upper": ci.iloc[:, 1],
        }
    )

    # Calculate significance stars
    def get_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.10:
            return "†"
        else:
            return ""

    summary["sig"] = summary["p_value"].apply(get_stars)

    # Round for readability
    summary["coef"] = summary["coef"].round(4)
    summary["se"] = summary["se"].round(4)
    summary["t_stat"] = summary["t_stat"].round(2)
    summary["p_value"] = summary["p_value"].round(4)
    summary["95%_CI_lower"] = summary["95%_CI_lower"].round(4)
    summary["95%_CI_upper"] = summary["95%_CI_upper"].round(4)

    logger.info("DiD summary table created with %d coefficients", len(summary))

    return summary


def dynamic_did(
    panel_df: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    entity_col: str,
    time_col: str,
    treatment_date: str,
    n_leads: int = 4,
    n_lags: int = 4,
    controls: list[str] | None = None,
) -> tuple[pd.DataFrame, Any]:
    """
    Dynamic DiD with event study leads and lags.

    Creates indicators for each period relative to treatment:
        treat × (t-4), treat × (t-3), ..., treat × t, ..., treat × (t+3), treat × (t+4)

    Useful for testing parallel trends and dynamic treatment effects.

    Args:
        panel_df: Panel data
        outcome: Dependent variable
        treatment_col: Treatment indicator
        entity_col: Entity identifier
        time_col: Time identifier (must be datetime or period)
        treatment_date: Date of treatment (str or pd.Timestamp)
        n_leads: Number of pre-treatment periods to include
        n_lags: Number of post-treatment periods to include
        controls: Control variables

    Returns:
        (coefficients_df, regression_result)
            coefficients_df: DataFrame with event time coefficients
            regression_result: Full regression output

    Example:
        >>> coefs, result = dynamic_did(
        ...     panel_df,
        ...     outcome='return',
        ...     treatment_col='treated',
        ...     entity_col='cik',
        ...     time_col='quarter',
        ...     treatment_date='2020Q1',
        ...     n_leads=4,
        ...     n_lags=4
        ... )
        >>> # Plot event study
        >>> plt.plot(coefs['event_time'], coefs['coef'])
    """
    df = panel_df.copy()

    # Convert treatment date
    treatment_date = pd.Period(treatment_date, freq="Q")

    # Convert time column to period if needed
    if not isinstance(df[time_col].iloc[0], pd.Period):
        df[time_col] = pd.PeriodIndex(df[time_col], freq="Q")

    # Calculate event time (periods since treatment)
    df["event_time"] = (df[time_col] - treatment_date).apply(lambda x: x.n)

    # Create lead/lag indicators
    for k in range(-n_leads, n_lags + 1):
        if k == -1:  # Omit t=-1 as reference period
            continue
        indicator = (df["event_time"] == k).astype(float)
        # Use 'm' for minus, 'p' for plus to avoid formula parsing issues
        var_name = f"treat_t{'m' if k < 0 else 'p'}{abs(k)}"
        df[var_name] = df[treatment_col] * indicator

    # Build formula
    lead_lag_vars = [
        f"treat_t{'m' if k < 0 else 'p'}{abs(k)}" for k in range(-n_leads, n_lags + 1) if k != -1
    ]
    formula = f"{outcome} ~ " + " + ".join(lead_lag_vars)

    if controls:
        formula += " + " + " + ".join(controls)

    formula += " + EntityEffects + TimeEffects"

    # Set index and estimate
    # Convert Period index to datetime for linearmodels compatibility
    df_indexed = df.copy()
    if isinstance(df_indexed[time_col].iloc[0], pd.Period):
        df_indexed[time_col] = df_indexed[time_col].dt.to_timestamp()

    df_indexed = df_indexed.set_index([entity_col, time_col])

    model = PanelOLS.from_formula(formula, data=df_indexed)
    result = model.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)

    # Extract lead/lag coefficients
    coef_data = []
    for k in range(-n_leads, n_lags + 1):
        if k == -1:
            # Reference period: coefficient = 0 by construction
            coef_data.append(
                {
                    "event_time": k,
                    "coef": 0.0,
                    "se": 0.0,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "95%_CI_lower": 0.0,
                    "95%_CI_upper": 0.0,
                }
            )
        else:
            var_name = f"treat_t{'m' if k < 0 else 'p'}{abs(k)}"
            ci = result.conf_int().loc[var_name]
            coef_data.append(
                {
                    "event_time": k,
                    "coef": result.params[var_name],
                    "se": result.std_errors[var_name],
                    "t_stat": result.tstats[var_name],
                    "p_value": result.pvalues[var_name],
                    "95%_CI_lower": ci.iloc[0],
                    "95%_CI_upper": ci.iloc[1],
                }
            )

    coef_df = pd.DataFrame(coef_data).sort_values("event_time")

    logger.info("Dynamic DiD complete: %d lead/lag coefficients estimated", len(coef_df))

    return coef_df, result


if __name__ == "__main__":  # pragma: no cover
    # Demo with synthetic data
    logger.info("DiD Demo with Synthetic Data")

    np.random.seed(42)

    # Create panel: 100 firms, 20 quarters (2018Q1-2022Q4)
    n_firms = 100
    n_quarters = 20
    quarters = pd.period_range("2018Q1", periods=n_quarters, freq="Q")

    panel_data = []
    for firm_id in range(n_firms):
        # Treatment: 50 firms treated in 2020Q1
        treated = 1 if firm_id < 50 else 0

        # Firm-specific intercept
        alpha_i = np.random.normal(0.02, 0.01)

        for quarter in quarters:
            # Post indicator
            post = 1 if quarter >= pd.Period("2020Q1", freq="Q") else 0

            # True treatment effect: -3% for treated firms after 2020Q1
            treatment_effect = -0.03 if (treated == 1 and post == 1) else 0.0

            # Generate return
            ret = alpha_i + 0.01 * post + treatment_effect + np.random.normal(0, 0.05)

            panel_data.append(
                {
                    "cik": firm_id,
                    "quarter": quarter.to_timestamp(),
                    "treated": treated,
                    "post_cecl": post,
                    "ret_next_quarter": ret,
                    "log_mcap": np.random.normal(8, 1),  # Control variable
                }
            )

    panel_df = pd.DataFrame(panel_data)

    # Prepare DiD data
    did_df = prepare_did_data(panel_df)

    # Run DiD regression
    result = run_did_regression(did_df, outcome="ret_next_quarter", controls=["log_mcap"])

    # Summary table
    summary = did_summary_table(result)
    print("\nDiD Summary Table:")
    print(summary)

    print("\nKey Finding:")
    print(f"DiD Coefficient (treat_post): {summary.loc['treat_post', 'coef']:.4f}")
    print(f"Standard Error: {summary.loc['treat_post', 'se']:.4f}")
    print(f"t-statistic: {summary.loc['treat_post', 't_stat']:.2f}")
    print(f"p-value: {summary.loc['treat_post', 'p_value']:.4f}")

    # Dynamic DiD
    print("\n" + "=" * 60)
    print("Dynamic DiD (Event Study)")
    print("=" * 60)

    coef_df, dyn_result = dynamic_did(
        panel_df,
        outcome="ret_next_quarter",
        treatment_col="treated",
        entity_col="cik",
        time_col="quarter",
        treatment_date="2020Q1",
        n_leads=4,
        n_lags=4,
        controls=["log_mcap"],
    )

    print("\nEvent Study Coefficients:")
    print(coef_df.to_string(index=False))
