"""
Validate CNOI construct against external benchmarks.

This module tests three types of validity for the CNOI index:
1. Convergent validity: CNOI correlates with established readability metrics
2. Discriminant validity: CNOI captures unique variance beyond simple readability
3. Predictive validity: CNOI predicts outcomes better than alternatives

References:
    - Campbell & Fiske (1959) - Convergent and discriminant validity
    - Cronbach & Meehl (1955) - Construct validity in psychological tests
    - Li (2008) - Readability and stock returns (JAR)
    - Loughran & McDonald (2014) - Measuring readability in financial disclosures
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def compute_cnoi_readability_correlations(
    cnoi_df: pd.DataFrame, readability_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Test convergent validity: CNOI should correlate with readability metrics.

    Expected correlations (based on construct alignment):
        - CNOI vs. Fog Index: ρ > 0.4 (both measure opacity)
        - CNOI vs. Flesch Ease: ρ < -0.4 (inverse - Flesch measures clarity)
        - CNOI vs. FK Grade: ρ > 0.4 (both increase with complexity)

    Moderate correlations (0.4-0.6) suggest CNOI captures similar constructs
    but is not redundant with simple readability.

    Args:
        cnoi_df: DataFrame with ['cik', 'filing_date', 'CNOI']
        readability_df: DataFrame with ['cik', 'filing_date', 'fog_index',
                       'flesch_ease', 'fk_grade', 'smog', etc.]

    Returns:
        Summary DataFrame with correlations and significance tests:
            - Metric: Name of readability metric
            - Correlation with CNOI: Pearson correlation coefficient
            - p-value: Two-tailed significance test
            - Significant: Whether p < 0.05
            - N: Sample size

    Examples:
        >>> cnoi_df = pd.DataFrame({
        ...     'cik': ['001', '002', '003'],
        ...     'filing_date': ['2023-12-31'] * 3,
        ...     'CNOI': [10.0, 15.0, 20.0]
        ... })
        >>> readability_df = pd.DataFrame({
        ...     'cik': ['001', '002', '003'],
        ...     'filing_date': ['2023-12-31'] * 3,
        ...     'fog_index': [12.0, 14.0, 16.0],
        ...     'flesch_ease': [60.0, 50.0, 40.0]
        ... })
        >>> summary = compute_cnoi_readability_correlations(cnoi_df, readability_df)
        >>> assert summary.loc[summary['Metric'] == 'fog_index', 'Correlation with CNOI'].values[0] > 0

    Notes:
        Uses Pearson correlation with two-tailed t-test for significance.
        Requires at least 3 observations for valid correlation.
    """
    # Merge datasets
    merged = cnoi_df.merge(readability_df, on=["cik", "filing_date"], how="inner")

    if len(merged) < 3:
        warnings.warn(
            f"Only {len(merged)} observations after merge. Need ≥3 for correlation.", stacklevel=2
        )
        return pd.DataFrame(
            columns=["Metric", "Correlation with CNOI", "p-value", "Significant", "N"]
        )

    # Readability metrics to correlate
    metrics = [
        "fog_index",
        "flesch_ease",
        "fk_grade",
        "smog",
        "word_count",
        "complex_word_pct",
        "avg_words_per_sentence",
    ]

    # Filter to available columns
    available_metrics = [m for m in metrics if m in merged.columns]

    if len(available_metrics) == 0:
        warnings.warn("No readability metrics found in merged DataFrame.", stacklevel=2)
        return pd.DataFrame(
            columns=["Metric", "Correlation with CNOI", "p-value", "Significant", "N"]
        )

    results = []
    len(merged)

    for metric in available_metrics:
        # Remove NaN values for this metric
        valid_data = merged[["CNOI", metric]].dropna()

        if len(valid_data) < 3:
            continue

        # Compute Pearson correlation
        r, p_val = stats.pearsonr(valid_data["CNOI"], valid_data[metric])

        results.append(
            {
                "Metric": metric,
                "Correlation with CNOI": r,
                "p-value": p_val,
                "Significant": p_val < 0.05,
                "N": len(valid_data),
            }
        )

    summary = pd.DataFrame(results)

    # Sort by absolute correlation (strongest first)
    if len(summary) > 0:
        summary["abs_corr"] = summary["Correlation with CNOI"].abs()
        summary = summary.sort_values("abs_corr", ascending=False)
        summary = summary.drop(columns=["abs_corr"])

    return summary


def dimension_contribution_analysis(
    cnoi_df: pd.DataFrame, dimensions: list[str] | None = None
) -> pd.DataFrame:
    """
    Analyze which CNOI dimensions contribute most to total score variance.

    CNOI formula:
        CNOI = 0.2×D + 0.2×G + 0.2×R + 0.1×J + 0.1×T + 0.1×S + 0.1×X

    This analysis:
    1. Correlates each dimension with total CNOI
    2. Computes variance explained (R²)
    3. Compares to assigned weights

    Interpretation:
        - High R² means dimension drives CNOI variation
        - Low R² suggests dimension is constant across banks

    Args:
        cnoi_df: DataFrame with CNOI and dimension columns
        dimensions: List of dimension column names
                   Default: ['D', 'G', 'R', 'J', 'T', 'S', 'X']

    Returns:
        DataFrame with:
            - Dimension: Column name
            - Correlation: Pearson r with CNOI
            - Variance Explained (R²): r²
            - Weight in CNOI: Assigned weight from formula
            - Mean: Average value of dimension
            - Std Dev: Standard deviation

    Examples:
        >>> cnoi_df = pd.DataFrame({
        ...     'D': [10, 15, 20],
        ...     'G': [12, 14, 16],
        ...     'R': [8, 12, 16],
        ...     'J': [9, 11, 13],
        ...     'T': [7, 8, 9],
        ...     'S': [6, 10, 14],
        ...     'X': [5, 7, 9],
        ...     'CNOI': [10.0, 13.0, 16.0]
        ... })
        >>> summary = dimension_contribution_analysis(cnoi_df)
        >>> assert 'Variance Explained (R²)' in summary.columns
    """
    if dimensions is None:
        dimensions = ["D", "G", "R", "J", "T", "S", "X"]

    # Filter to available dimensions
    available_dims = [d for d in dimensions if d in cnoi_df.columns]

    if "CNOI" not in cnoi_df.columns:
        raise ValueError("cnoi_df must contain 'CNOI' column")

    if len(available_dims) == 0:
        raise ValueError(f"No dimension columns found in cnoi_df. Expected: {dimensions}")

    # Dimension weights (from CNOI formula)
    weights = {"D": 0.2, "G": 0.2, "R": 0.2, "J": 0.1, "T": 0.1, "S": 0.1, "X": 0.1}

    results = []

    for dim in available_dims:
        # Remove NaN values
        valid_data = cnoi_df[[dim, "CNOI"]].dropna()

        if len(valid_data) < 3:
            continue

        # Correlation with CNOI
        r, _ = stats.pearsonr(valid_data[dim], valid_data["CNOI"])

        # Variance explained
        r_squared = r**2

        results.append(
            {
                "Dimension": dim,
                "Correlation": r,
                "Variance Explained (R²)": r_squared,
                "Weight in CNOI": weights.get(dim, 0.0),
                "Mean": cnoi_df[dim].mean(),
                "Std Dev": cnoi_df[dim].std(),
            }
        )

    summary = pd.DataFrame(results)

    # Sort by variance explained (highest first)
    if len(summary) > 0:
        summary = summary.sort_values("Variance Explained (R²)", ascending=False)

    return summary


def horse_race_regression(
    outcome_df: pd.DataFrame,
    cnoi_df: pd.DataFrame,
    readability_df: pd.DataFrame,
    outcome_col: str = "ret_next_quarter",
    readability_col: str = "fog_index",
) -> tuple[pd.DataFrame, dict]:
    """
    Test discriminant validity: Does CNOI predict outcomes beyond readability?

    Horse-race regression strategy:
        Model 1: Y = α + β1·CNOI + ε
        Model 2: Y = α + β2·ReadabilityMetric + ε
        Model 3: Y = α + β3·CNOI + β4·ReadabilityMetric + ε  (horse race)

    If β3 remains significant in Model 3 → CNOI has incremental predictive power
    beyond simple readability.

    Args:
        outcome_df: DataFrame with outcome variable (e.g., returns, volatility)
                   Must have ['cik', 'filing_date', outcome_col]
        cnoi_df: DataFrame with CNOI scores
        readability_df: DataFrame with readability metrics
        outcome_col: Name of outcome variable
        readability_col: Which readability metric to use (default: 'fog_index')

    Returns:
        Tuple of:
            - summary: DataFrame comparing model statistics
            - models: Dict with fitted model objects

    Examples:
        >>> outcome_df = pd.DataFrame({
        ...     'cik': ['001', '002', '003'],
        ...     'filing_date': ['2023-12-31'] * 3,
        ...     'ret_next_quarter': [0.05, -0.02, 0.03]
        ... })
        >>> cnoi_df = pd.DataFrame({
        ...     'cik': ['001', '002', '003'],
        ...     'filing_date': ['2023-12-31'] * 3,
        ...     'CNOI': [10.0, 15.0, 20.0]
        ... })
        >>> readability_df = pd.DataFrame({
        ...     'cik': ['001', '002', '003'],
        ...     'filing_date': ['2023-12-31'] * 3,
        ...     'fog_index': [12.0, 14.0, 16.0]
        ... })
        >>> summary, models = horse_race_regression(
        ...     outcome_df, cnoi_df, readability_df
        ... )
        >>> assert 'Model' in summary.columns
        >>> assert 'CNOI_coef' in summary.columns

    Notes:
        Uses OLS regression with robust standard errors.
        Tests incremental R² using F-test for nested models.
    """
    # Merge all datasets
    merged = outcome_df.merge(cnoi_df, on=["cik", "filing_date"], how="inner")
    merged = merged.merge(readability_df, on=["cik", "filing_date"], how="inner")

    # Check for required columns
    if outcome_col not in merged.columns:
        raise ValueError(f"Outcome column '{outcome_col}' not found in merged data")
    if "CNOI" not in merged.columns:
        raise ValueError("CNOI column not found in merged data")
    if readability_col not in merged.columns:
        raise ValueError(f"Readability column '{readability_col}' not found in merged data")

    # Remove NaN values
    required_cols = [outcome_col, "CNOI", readability_col]
    merged = merged[required_cols].dropna()

    if len(merged) < 3:
        raise ValueError(f"Insufficient data after merge: {len(merged)} observations")

    y = merged[outcome_col]

    # Model 1: CNOI only
    X1 = sm.add_constant(merged["CNOI"])
    model1 = sm.OLS(y, X1).fit(cov_type="HC3")  # Robust SEs

    # Model 2: Readability metric only
    X2 = sm.add_constant(merged[readability_col])
    model2 = sm.OLS(y, X2).fit(cov_type="HC3")

    # Model 3: Both (horse race)
    X3 = sm.add_constant(merged[["CNOI", readability_col]])
    model3 = sm.OLS(y, X3).fit(cov_type="HC3")

    # Create summary table
    summary_data = []

    # Model 1
    summary_data.append(
        {
            "Model": "CNOI only",
            "CNOI_coef": model1.params["CNOI"],
            "CNOI_tstat": model1.tvalues["CNOI"],
            "CNOI_pval": model1.pvalues["CNOI"],
            f"{readability_col}_coef": np.nan,
            f"{readability_col}_tstat": np.nan,
            f"{readability_col}_pval": np.nan,
            "R²": model1.rsquared,
            "Adj_R²": model1.rsquared_adj,
            "N": int(model1.nobs),
        }
    )

    # Model 2
    summary_data.append(
        {
            "Model": f"{readability_col} only",
            "CNOI_coef": np.nan,
            "CNOI_tstat": np.nan,
            "CNOI_pval": np.nan,
            f"{readability_col}_coef": model2.params[readability_col],
            f"{readability_col}_tstat": model2.tvalues[readability_col],
            f"{readability_col}_pval": model2.pvalues[readability_col],
            "R²": model2.rsquared,
            "Adj_R²": model2.rsquared_adj,
            "N": int(model2.nobs),
        }
    )

    # Model 3 (Horse Race)
    summary_data.append(
        {
            "Model": f"CNOI + {readability_col} (Horse Race)",
            "CNOI_coef": model3.params["CNOI"],
            "CNOI_tstat": model3.tvalues["CNOI"],
            "CNOI_pval": model3.pvalues["CNOI"],
            f"{readability_col}_coef": model3.params[readability_col],
            f"{readability_col}_tstat": model3.tvalues[readability_col],
            f"{readability_col}_pval": model3.pvalues[readability_col],
            "R²": model3.rsquared,
            "Adj_R²": model3.rsquared_adj,
            "N": int(model3.nobs),
        }
    )

    summary = pd.DataFrame(summary_data)

    models = {"cnoi_only": model1, "readability_only": model2, "horse_race": model3}

    return summary, models


def compute_incremental_r_squared(model_restricted, model_full) -> dict[str, float]:
    """
    F-test for incremental R² from adding CNOI to readability-only model.

    Tests H0: Adding CNOI does not increase R²

    Args:
        model_restricted: Model without CNOI (e.g., readability only)
        model_full: Model with CNOI (e.g., CNOI + readability)

    Returns:
        Dictionary with:
            - incremental_r2: R²_full - R²_restricted
            - f_stat: F-statistic for nested model test
            - p_value: Significance of incremental R²
            - df_num: Degrees of freedom (numerator)
            - df_denom: Degrees of freedom (denominator)

    Notes:
        F = [(R²_full - R²_restricted) / q] / [(1 - R²_full) / (n - k - 1)]
        where q = # of additional variables, n = sample size, k = # params in full model
    """
    r2_restricted = model_restricted.rsquared
    r2_full = model_full.rsquared

    n = int(model_full.nobs)
    k_restricted = len(model_restricted.params)
    k_full = len(model_full.params)

    q = k_full - k_restricted  # Number of additional variables

    if q <= 0:
        raise ValueError("Full model must have more parameters than restricted model")

    # Incremental R²
    incremental_r2 = r2_full - r2_restricted

    # F-statistic
    numerator = incremental_r2 / q
    denominator = (1 - r2_full) / (n - k_full)

    f_stat = numerator / denominator if denominator > 0 else np.nan

    # p-value from F distribution
    df_num = q
    df_denom = n - k_full

    if not np.isnan(f_stat):
        p_value = 1 - stats.f.cdf(f_stat, df_num, df_denom)
    else:
        p_value = np.nan

    return {
        "incremental_r2": incremental_r2,
        "f_stat": f_stat,
        "p_value": p_value,
        "df_num": df_num,
        "df_denom": df_denom,
    }


def compute_partial_correlations(
    data: pd.DataFrame, y_col: str, x1_col: str, x2_col: str
) -> dict[str, float]:
    """
    Compute partial correlations to isolate CNOI effect.

    Partial correlation of Y with X1 controlling for X2:
        ρ_Y,X1|X2 = (ρ_Y,X1 - ρ_Y,X2 × ρ_X1,X2) / sqrt((1-ρ²_Y,X2)(1-ρ²_X1,X2))

    Args:
        data: DataFrame with all variables
        y_col: Outcome variable (e.g., returns)
        x1_col: Primary predictor (e.g., CNOI)
        x2_col: Control variable (e.g., fog_index)

    Returns:
        Dictionary with:
            - partial_corr_x1: Partial correlation of Y with X1 | X2
            - partial_corr_x2: Partial correlation of Y with X2 | X1
            - zero_order_corr_x1: Regular correlation Y with X1
            - zero_order_corr_x2: Regular correlation Y with X2

    Examples:
        >>> data = pd.DataFrame({
        ...     'ret': [0.05, -0.02, 0.03, 0.01],
        ...     'CNOI': [10, 15, 20, 12],
        ...     'fog': [12, 14, 16, 13]
        ... })
        >>> partial = compute_partial_correlations(data, 'ret', 'CNOI', 'fog')
        >>> assert 'partial_corr_x1' in partial
    """
    # Check columns exist
    for col in [y_col, x1_col, x2_col]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    # Remove NaN
    clean_data = data[[y_col, x1_col, x2_col]].dropna()

    if len(clean_data) < 3:
        raise ValueError(f"Insufficient data: {len(clean_data)} observations")

    # Zero-order correlations
    r_yx1, _ = stats.pearsonr(clean_data[y_col], clean_data[x1_col])
    r_yx2, _ = stats.pearsonr(clean_data[y_col], clean_data[x2_col])
    r_x1x2, _ = stats.pearsonr(clean_data[x1_col], clean_data[x2_col])

    # Partial correlation: Y with X1 | X2
    numerator_x1 = r_yx1 - r_yx2 * r_x1x2
    denominator_x1 = np.sqrt((1 - r_yx2**2) * (1 - r_x1x2**2))
    partial_x1 = numerator_x1 / denominator_x1 if denominator_x1 > 0 else np.nan

    # Partial correlation: Y with X2 | X1
    numerator_x2 = r_yx2 - r_yx1 * r_x1x2
    denominator_x2 = np.sqrt((1 - r_yx1**2) * (1 - r_x1x2**2))
    partial_x2 = numerator_x2 / denominator_x2 if denominator_x2 > 0 else np.nan

    return {
        "partial_corr_x1": partial_x1,  # CNOI | readability
        "partial_corr_x2": partial_x2,  # Readability | CNOI
        "zero_order_corr_x1": r_yx1,
        "zero_order_corr_x2": r_yx2,
        "corr_x1_x2": r_x1x2,
    }
