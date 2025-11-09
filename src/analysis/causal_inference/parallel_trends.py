"""
Test parallel trends assumption for DiD validity.

Key idea: Treated and control groups must have parallel outcome trends BEFORE treatment.
If pre-trends diverge, DiD is biased.

References:
- Angrist & Pischke (2009) - Mostly Harmless Econometrics
- Kahn-Lang & Lang (2020) - The promise and pitfalls of differences-in-differences
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_parallel_trends(
    panel_df: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    time_col: str,
    entity_col: str,
    pre_period_end: str,
    min_periods: int = 3,
) -> dict[str, Any]:
    """
    Test if treated/control groups have parallel trends pre-treatment.

    Method:
        1. Subset to pre-period only
        2. Run regression: Y_it = α + β·Treat_i + Σ(γ_t·Treat_i × Period_t) + ε
        3. Test H0: γ_t = 0 for all pre-periods (F-test)

    If H0 rejected → parallel trends violated

    Args:
        panel_df: Panel DataFrame
        outcome: Dependent variable
        treatment_col: Treatment indicator column
        time_col: Time period column
        entity_col: Entity identifier
        pre_period_end: End of pre-treatment period (inclusive)
        min_periods: Minimum number of pre-periods required

    Returns:
        {
            'f_stat': float,
            'f_pvalue': float,
            'violated': bool,  # True if p < 0.05
            'n_periods': int,
            'n_treated': int,
            'n_control': int,
            'interaction_coefs': DataFrame with coefficients on Treat×Period terms
        }

    Example:
        >>> result = test_parallel_trends(
        ...     panel_df,
        ...     outcome='return',
        ...     treatment_col='treated',
        ...     time_col='quarter',
        ...     entity_col='cik',
        ...     pre_period_end='2019Q4'
        ... )
        >>> print(f"Parallel trends violated: {result['violated']}")
    """
    # Convert pre_period_end to Period if needed
    if isinstance(pre_period_end, str):
        pre_period_end = pd.Period(pre_period_end, freq="Q")

    # Filter to pre-period
    df = panel_df.copy()

    # Convert time column to Period if needed
    if not isinstance(df[time_col].iloc[0], pd.Period):
        df[time_col] = pd.PeriodIndex(df[time_col], freq="Q")

    pre_df = df[df[time_col] <= pre_period_end].copy()

    if len(pre_df) == 0:
        raise ValueError(f"No data found before {pre_period_end}")

    # Count periods
    periods = pre_df[time_col].unique()
    n_periods = len(periods)

    if n_periods < min_periods:
        raise ValueError(f"Need at least {min_periods} pre-periods, only have {n_periods}")

    # Count treated/control
    n_treated = pre_df[pre_df[treatment_col] == 1][entity_col].nunique()
    n_control = pre_df[pre_df[treatment_col] == 0][entity_col].nunique()

    logger.info(
        "Testing parallel trends: %d pre-periods, %d treated, %d control",
        n_periods,
        n_treated,
        n_control,
    )

    # Create period dummies (excluding first period as reference)
    sorted_periods = sorted(periods)

    # Create Treat × Period interactions (excluding reference period)
    interaction_terms = []
    for i, period in enumerate(sorted_periods[1:], start=1):  # Skip first period
        period_dummy = (pre_df[time_col] == period).astype(float)
        interaction = pre_df[treatment_col] * period_dummy
        var_name = f"treat_x_period_{i}"
        pre_df[var_name] = interaction
        interaction_terms.append(var_name)

    # Create time period categorical for formula
    pre_df["time_period_cat"] = pre_df[time_col].astype(str)

    # Build regression formula
    formula = f"{outcome} ~ C(time_period_cat) + {treatment_col}"
    if interaction_terms:
        formula += " + " + " + ".join(interaction_terms)

    # Run OLS
    try:
        model = smf.ols(formula, data=pre_df).fit(
            cov_type="cluster", cov_kwds={"groups": pre_df[entity_col]}
        )

        # Extract interaction coefficients
        interaction_coefs = []
        for term in interaction_terms:
            if term in model.params.index:
                interaction_coefs.append(
                    {
                        "term": term,
                        "coef": model.params[term],
                        "se": model.bse[term],
                        "t_stat": model.tvalues[term],
                        "p_value": model.pvalues[term],
                    }
                )

        coef_df = pd.DataFrame(interaction_coefs)

        # F-test: All interaction coefficients jointly = 0
        if len(interaction_terms) > 0:
            f_test = model.f_test(interaction_terms)
            # Handle different statsmodels versions
            if hasattr(f_test.fvalue, "__getitem__"):
                # Array-like (older versions)
                f_stat = f_test.fvalue[0][0] if f_test.fvalue.ndim > 1 else f_test.fvalue[0]
            else:
                # Scalar (newer versions)
                f_stat = float(f_test.fvalue)
            f_pvalue = float(f_test.pvalue)
        else:
            f_stat = np.nan
            f_pvalue = np.nan

        violated = f_pvalue < 0.05 if not np.isnan(f_pvalue) else False

        logger.info(
            "Parallel trends test: F=%.2f, p=%.4f, violated=%s",
            f_stat,
            f_pvalue,
            violated,
        )

        return {
            "f_stat": f_stat,
            "f_pvalue": f_pvalue,
            "violated": violated,
            "n_periods": n_periods,
            "n_treated": n_treated,
            "n_control": n_control,
            "interaction_coefs": coef_df,
            "full_model": model,
        }

    except Exception as e:
        logger.error("Parallel trends test failed: %s", str(e))
        raise


def plot_parallel_trends(
    panel_df: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    time_col: str,
    treatment_date: str,
    figsize: tuple = (12, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize treated vs. control trends over time.

    Creates line plot:
        - X-axis: Quarter
        - Y-axis: Mean outcome
        - Two lines: Treated group (blue), Control group (red)
        - Vertical line at treatment date

    Should show parallel lines pre-treatment, divergence post-treatment.

    Args:
        panel_df: Panel DataFrame
        outcome: Variable to plot (e.g., 'return')
        treatment_col: Treatment indicator
        time_col: Time period column
        treatment_date: When treatment occurred (for vertical line)
        figsize: Figure size
        save_path: Optional path to save figure

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_parallel_trends(
        ...     panel_df,
        ...     outcome='return',
        ...     treatment_col='treated',
        ...     time_col='quarter',
        ...     treatment_date='2020Q1'
        ... )
        >>> plt.show()
    """
    df = panel_df.copy()

    # Convert time to Period if needed
    if not isinstance(df[time_col].iloc[0], pd.Period):
        df[time_col] = pd.PeriodIndex(df[time_col], freq="Q")

    # Convert treatment date
    treatment_date = pd.Period(treatment_date, freq="Q")

    # Compute group means by time
    treated = df[df[treatment_col] == 1].groupby(time_col)[outcome].mean()
    control = df[df[treatment_col] == 0].groupby(time_col)[outcome].mean()

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot lines
    ax.plot(
        treated.index.to_timestamp(),
        treated.values,
        label="Treated (CECL adopters)",
        marker="o",
        linewidth=2,
        markersize=6,
        color="steelblue",
    )
    ax.plot(
        control.index.to_timestamp(),
        control.values,
        label="Control",
        marker="s",
        linewidth=2,
        markersize=6,
        color="coral",
    )

    # Add vertical line at treatment
    ax.axvline(
        treatment_date.to_timestamp(),
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
        label=f"Treatment ({treatment_date})",
    )

    # Styling
    ax.set_xlabel("Quarter", fontsize=12)
    ax.set_ylabel(f"Mean {outcome}", fontsize=12)
    ax.set_title("Parallel Trends: Treated vs. Control Groups", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Rotate x-axis labels
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Parallel trends plot saved to %s", save_path)

    return fig


def placebo_test(
    panel_df: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    entity_col: str,
    time_col: str,
    true_treatment_date: str,
    fake_treatment_date: str,
    controls: list | None = None,
) -> dict[str, Any]:
    """
    Placebo DiD test: Run DiD with fake treatment date in pre-period.

    Should find NO effect (δ ≈ 0) if parallel trends hold.

    Args:
        panel_df: Panel DataFrame
        outcome: Dependent variable
        treatment_col: Treatment indicator
        entity_col: Entity identifier
        time_col: Time column
        true_treatment_date: Actual treatment date (to filter data before this)
        fake_treatment_date: Fake treatment date for placebo test
        controls: Control variables

    Returns:
        {
            'did_coef': float,
            'se': float,
            't_stat': float,
            'p_value': float,
            'passed': bool  # True if p > 0.10 (no fake effect)
        }

    Example:
        >>> result = placebo_test(
        ...     panel_df,
        ...     outcome='return',
        ...     treatment_col='treated',
        ...     entity_col='cik',
        ...     time_col='quarter',
        ...     true_treatment_date='2020Q1',
        ...     fake_treatment_date='2019Q3'
        ... )
        >>> print(f"Placebo test passed: {result['passed']}")
    """
    from src.analysis.causal_inference.difference_in_differences import (
        prepare_did_data,
        run_did_regression,
    )

    # Convert dates
    true_treatment = pd.Period(true_treatment_date, freq="Q")
    fake_treatment = pd.Period(fake_treatment_date, freq="Q")

    if fake_treatment >= true_treatment:
        raise ValueError("Fake treatment date must be before true treatment date")

    # Filter to pre-treatment period only
    df = panel_df.copy()
    if not isinstance(df[time_col].iloc[0], pd.Period):
        df[time_col] = pd.PeriodIndex(df[time_col], freq="Q")

    pre_df = df[df[time_col] < true_treatment].copy()

    if len(pre_df) == 0:
        raise ValueError(f"No data before {true_treatment}")

    # Create fake post indicator
    pre_df["post_fake"] = (pre_df[time_col] >= fake_treatment).astype(float)

    # Convert Period to timestamp for DiD regression
    if isinstance(pre_df[time_col].iloc[0], pd.Period):
        pre_df[time_col] = pre_df[time_col].dt.to_timestamp()

    # Prepare DiD data with fake post
    pre_df["treated"] = pre_df[treatment_col]
    did_df = prepare_did_data(
        pre_df,
        treatment_col="treated",
        post_col="post_fake",
        outcome_col=outcome,
        controls=controls,
    )

    # Run DiD regression
    result = run_did_regression(
        did_df,
        outcome=outcome,
        entity_col=entity_col,
        time_col=time_col,
        controls=controls,
        cluster_entity=True,
        cluster_time=True,
        entity_fe=True,
        time_fe=True,
    )

    # Extract DiD coefficient
    did_coef = result.params["treat_post"]
    se = result.std_errors["treat_post"]
    t_stat = result.tstats["treat_post"]
    p_value = result.pvalues["treat_post"]

    # Test passes if p > 0.10 (no significant fake effect)
    passed = p_value > 0.10

    logger.info(
        "Placebo test (fake treatment=%s): coef=%.4f, t=%.2f, p=%.4f, passed=%s",
        fake_treatment,
        did_coef,
        t_stat,
        p_value,
        passed,
    )

    return {
        "did_coef": did_coef,
        "se": se,
        "t_stat": t_stat,
        "p_value": p_value,
        "passed": passed,
        "fake_treatment_date": str(fake_treatment),
        "n_obs": len(did_df),
    }


def granger_causality_test(
    panel_df: pd.DataFrame,
    outcome: str,
    treatment_col: str,
    time_col: str,
    entity_col: str,
    max_lags: int = 4,
) -> dict[str, Any]:
    """
    Test if treatment Granger-causes outcome (or vice versa).

    Useful for checking reverse causality concerns.

    Args:
        panel_df: Panel DataFrame
        outcome: Dependent variable
        treatment_col: Treatment variable
        time_col: Time column
        entity_col: Entity identifier
        max_lags: Maximum number of lags to test

    Returns:
        {
            'outcome_granger_causes_treatment': {'f_stat': float, 'p_value': float},
            'treatment_granger_causes_outcome': {'f_stat': float, 'p_value': float}
        }
    """

    # This requires time-series data per entity
    # Implementation would require reshaping panel to wide format
    # Placeholder for future implementation

    logger.warning("Granger causality test not yet implemented for panel data")

    return {
        "outcome_granger_causes_treatment": {"f_stat": np.nan, "p_value": np.nan},
        "treatment_granger_causes_outcome": {"f_stat": np.nan, "p_value": np.nan},
    }


if __name__ == "__main__":  # pragma: no cover
    # Demo with synthetic data
    logger.info("Parallel Trends Demo")

    np.random.seed(42)

    # Create panel with parallel trends
    n_firms = 100
    n_quarters = 20
    quarters = pd.period_range("2018Q1", periods=n_quarters, freq="Q")

    panel_data = []
    for firm_id in range(n_firms):
        treated = 1 if firm_id < 50 else 0
        alpha_i = np.random.normal(0.02, 0.01)

        # Common time trend (same for both groups → parallel trends)
        time_trend = 0.001

        for i, quarter in enumerate(quarters):
            post = 1 if quarter >= pd.Period("2020Q1", freq="Q") else 0

            # Treatment effect only post-treatment
            treatment_effect = -0.03 if (treated == 1 and post == 1) else 0.0

            # Generate return with common trend
            ret = alpha_i + time_trend * i + treatment_effect + np.random.normal(0, 0.02)

            panel_data.append(
                {
                    "cik": firm_id,
                    "quarter": quarter.to_timestamp(),
                    "treated": treated,
                    "post_cecl": post,
                    "return": ret,
                }
            )

    panel_df = pd.DataFrame(panel_data)

    # Test parallel trends
    print("\n" + "=" * 60)
    print("Parallel Trends Test")
    print("=" * 60)

    result = check_parallel_trends(
        panel_df,
        outcome="return",
        treatment_col="treated",
        time_col="quarter",
        entity_col="cik",
        pre_period_end="2019Q4",
    )

    print(f"F-statistic: {result['f_stat']:.2f}")
    print(f"p-value: {result['f_pvalue']:.4f}")
    print(f"Parallel trends violated: {result['violated']}")
    print(f"\nPre-periods: {result['n_periods']}")
    print(f"Treated firms: {result['n_treated']}")
    print(f"Control firms: {result['n_control']}")

    # Plot
    fig = plot_parallel_trends(
        panel_df,
        outcome="return",
        treatment_col="treated",
        time_col="quarter",
        treatment_date="2020Q1",
    )
    plt.show()

    # Placebo test
    print("\n" + "=" * 60)
    print("Placebo Test")
    print("=" * 60)

    placebo_result = placebo_test(
        panel_df,
        outcome="return",
        treatment_col="treated",
        entity_col="cik",
        time_col="quarter",
        true_treatment_date="2020Q1",
        fake_treatment_date="2019Q3",
    )

    print(f"Fake DiD coefficient: {placebo_result['did_coef']:.4f}")
    print(f"Standard error: {placebo_result['se']:.4f}")
    print(f"t-statistic: {placebo_result['t_stat']:.2f}")
    print(f"p-value: {placebo_result['p_value']:.4f}")
    print(f"Test passed (no fake effect): {placebo_result['passed']}")
