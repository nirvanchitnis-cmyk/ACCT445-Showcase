"""
Wild cluster bootstrap for DiD estimation with few clusters.

References:
- Cameron, Gelbach & Miller (2008) - Bootstrap-based improvements for inference with clustered errors
- Roodman et al. (2019) - Fast and wild: Bootstrap inference in Stata using boottest
- MacKinnon & Webb (2017) - Wild bootstrap inference for wildly different cluster sizes
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.utils.logger import get_logger

logger = get_logger(__name__)


def wild_cluster_bootstrap(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    entity_id: str,
    time_id: str | None = None,
    cluster_var: str | None = None,
    controls: list[str] | None = None,
    n_boot: int = 999,
    seed: int = 42,
    weight_type: str = "rademacher",
) -> dict:
    """
    Wild cluster bootstrap for DiD with few clusters.

    Implements Cameron-Gelbach-Miller (2008) wild cluster bootstrap
    for two-way fixed effects DiD when the number of treated clusters
    is small (< 20-30).

    Args:
        data: Panel DataFrame with bank-quarter observations
        outcome: Dependent variable column name
        treatment: Treatment variable (DiD interaction: treated × post)
        entity_id: Cluster variable (e.g., 'cik' for bank ID)
        time_id: Time fixed effects variable (e.g., 'quarter')
        cluster_var: Variable to cluster on (default: entity_id)
        controls: List of control variables
        n_boot: Number of bootstrap replications (default 999)
        seed: Random seed for reproducibility
        weight_type: Bootstrap weight distribution ('rademacher', 'mammen', 'normal')

    Returns:
        {
            'coef': float,             # DiD coefficient
            'se_conventional': float,  # Conventional clustered SE
            'se_bootstrap': float,     # Bootstrap SE
            't_stat': float,           # Original t-statistic
            'p_conventional': float,   # Conventional p-value
            'p_bootstrap': float,      # Wild bootstrap p-value
            'p_bootstrap_sym': float,  # Symmetric two-tailed bootstrap p-value
            'ci_lower': float,         # 95% bootstrap CI lower bound
            'ci_upper': float,         # 95% bootstrap CI upper bound
            'n_clusters': int,         # Number of clusters
            'n_boot': int,             # Number of bootstrap replications
        }

    Example:
        >>> # Prepare DiD data
        >>> df['treat_post'] = df['treated'] * df['post_cecl']
        >>>
        >>> # Run wild bootstrap
        >>> result = wild_cluster_bootstrap(
        ...     data=df,
        ...     outcome='ret_next_quarter',
        ...     treatment='treat_post',
        ...     entity_id='cik',
        ...     time_id='quarter',
        ...     n_boot=999
        ... )
        >>> print(f\"DiD coef: {result['coef']:.4f}, WCB p-value: {result['p_bootstrap']:.4f}\")
    """
    if cluster_var is None:
        cluster_var = entity_id

    # Prepare regression variables
    y = data[outcome].astype(float).copy()
    X_vars = [treatment]
    if controls:
        X_vars.extend(controls)

    # Add fixed effects dummies
    X_df = data[X_vars].astype(float).copy()

    # Entity fixed effects
    entity_dummies = pd.get_dummies(data[entity_id], prefix="entity", drop_first=True).astype(float)
    X_df = pd.concat([X_df, entity_dummies], axis=1)

    # Time fixed effects (if provided)
    if time_id:
        time_dummies = pd.get_dummies(data[time_id], prefix="time", drop_first=True).astype(float)
        X_df = pd.concat([X_df, time_dummies], axis=1)

    # Add constant
    X_df = sm.add_constant(X_df)

    # Align y and X
    common_idx = y.index.intersection(X_df.index)
    y = y.loc[common_idx]
    X_df = X_df.loc[common_idx]

    # Step 1: Fit original model
    model = sm.OLS(y, X_df).fit()

    # Get treatment coefficient and t-stat
    X_df.columns.get_loc(treatment)
    coef_original = model.params[treatment]
    se_conventional = model.bse[treatment]
    t_original = model.tvalues[treatment]
    p_conventional = model.pvalues[treatment]

    # Get residuals
    residuals = model.resid

    # Get cluster IDs
    cluster_ids = data.loc[common_idx, cluster_var]
    unique_clusters = cluster_ids.unique()
    n_clusters = len(unique_clusters)

    logger.info(
        "Starting wild cluster bootstrap: %d clusters, %d replications, weight=%s",
        n_clusters,
        n_boot,
        weight_type,
    )

    # Step 2: Wild cluster bootstrap
    rng = np.random.default_rng(seed)
    bootstrap_t_stats = []

    for b in range(n_boot):
        # Generate cluster-level weights
        if weight_type == "rademacher":
            # Rademacher: +1 or -1 with equal probability
            weights_cluster = rng.choice([-1, 1], size=n_clusters)
        elif weight_type == "mammen":
            # Mammen (1993): two-point distribution
            (1 + np.sqrt(5)) / 2
            p = (np.sqrt(5) + 1) / (2 * np.sqrt(5))
            weights_cluster = rng.choice(
                [-(np.sqrt(5) - 1) / 2, (np.sqrt(5) + 1) / 2],
                size=n_clusters,
                p=[p, 1 - p],
            )
        elif weight_type == "normal":
            # Normal(0,1)
            weights_cluster = rng.standard_normal(n_clusters)
        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        # Map cluster weights to observations
        cluster_to_weight = dict(zip(unique_clusters, weights_cluster, strict=False))
        weights_obs = cluster_ids.map(cluster_to_weight).values

        # Create bootstrap residuals
        resid_boot = residuals * weights_obs

        # Create bootstrap outcome
        y_boot = model.fittedvalues + resid_boot

        # Fit bootstrap model
        try:
            model_boot = sm.OLS(y_boot, X_df).fit()
            t_boot = model_boot.tvalues[treatment]
            bootstrap_t_stats.append(t_boot)
        except Exception as e:
            logger.warning("Bootstrap iteration %d failed: %s", b, e)
            continue

    bootstrap_t_stats = np.array(bootstrap_t_stats)

    if len(bootstrap_t_stats) == 0:
        raise ValueError("All bootstrap iterations failed")

    # Step 3: Compute bootstrap p-value
    # Two approaches:
    # 1. Symmetric two-tailed: P(|t_boot| >= |t_obs|)
    # 2. One-sided: P(t_boot >= t_obs) for t_obs > 0, P(t_boot <= t_obs) for t_obs < 0

    # Symmetric two-tailed
    p_bootstrap_sym = (np.sum(np.abs(bootstrap_t_stats) >= np.abs(t_original)) + 1) / (n_boot + 1)

    # One-sided (asymmetric)
    if t_original > 0:
        p_bootstrap_one = (np.sum(bootstrap_t_stats >= t_original) + 1) / (n_boot + 1)
    else:
        p_bootstrap_one = (np.sum(bootstrap_t_stats <= t_original) + 1) / (n_boot + 1)

    # Two-tailed from one-sided
    p_bootstrap = 2 * min(p_bootstrap_one, 1 - p_bootstrap_one)

    # Step 4: Bootstrap confidence interval (percentile method)
    # Re-center bootstrap distribution around original estimate
    coef_boot = coef_original + (bootstrap_t_stats - t_original) * se_conventional
    ci_lower, ci_upper = np.percentile(coef_boot, [2.5, 97.5])

    # Bootstrap SE
    se_bootstrap = np.std(coef_boot, ddof=1)

    logger.info(
        "Wild bootstrap complete: coef=%.4f, p_conv=%.4f, p_boot=%.4f, n_clusters=%d",
        coef_original,
        p_conventional,
        p_bootstrap,
        n_clusters,
    )

    return {
        "coef": coef_original,
        "se_conventional": se_conventional,
        "se_bootstrap": se_bootstrap,
        "t_stat": t_original,
        "p_conventional": p_conventional,
        "p_bootstrap": p_bootstrap,
        "p_bootstrap_sym": p_bootstrap_sym,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_clusters": n_clusters,
        "n_boot": len(bootstrap_t_stats),
        "weight_type": weight_type,
    }


def compute_effective_clusters(
    data: pd.DataFrame,
    treatment_col: str,
    entity_col: str,
    time_col: str | None = None,
) -> dict:
    """
    Compute effective number of clusters for two-way clustering.

    Reports cluster counts to assess whether wild bootstrap is needed.
    Rule of thumb: If n_treated_clusters < 20, use wild bootstrap.

    Args:
        data: Panel DataFrame
        treatment_col: Treatment indicator (0/1)
        entity_col: Entity ID (e.g., bank CIK)
        time_col: Time ID (e.g., quarter)

    Returns:
        {
            'n_clusters_entity': int,      # Total entities
            'n_clusters_time': int,        # Total time periods
            'n_treated_entities': int,     # Treated entities
            'n_control_entities': int,     # Control entities
            'pct_treated': float,          # % entities treated
            'min_cluster_dim': int,        # Min(n_treated, n_control)
            'needs_wild_bootstrap': bool   # True if min < 20
        }

    Example:
        >>> cluster_info = compute_effective_clusters(
        ...     df, treatment_col='treated', entity_col='cik', time_col='quarter'
        ... )
        >>> if cluster_info['needs_wild_bootstrap']:
        ...     print("WARNING: Few clusters detected. Use wild bootstrap!")
    """
    # Entity clusters
    unique_entities = data[entity_col].nunique()

    # Treated vs. control entities
    entity_treatment = data.groupby(entity_col)[treatment_col].max()
    n_treated = (entity_treatment == 1).sum()
    n_control = (entity_treatment == 0).sum()

    pct_treated = n_treated / unique_entities if unique_entities > 0 else 0

    # Time clusters
    n_time = data[time_col].nunique() if time_col else 0

    # Minimum cluster dimension
    min_cluster_dim = min(n_treated, n_control)

    # Guideline: Need wild bootstrap if min cluster dim < 20
    needs_wild_bootstrap = min_cluster_dim < 20

    result = {
        "n_clusters_entity": unique_entities,
        "n_clusters_time": n_time,
        "n_treated_entities": n_treated,
        "n_control_entities": n_control,
        "pct_treated": pct_treated,
        "min_cluster_dim": min_cluster_dim,
        "needs_wild_bootstrap": needs_wild_bootstrap,
    }

    logger.info(
        "Cluster counts: %d total entities (%d treated, %d control), %d time periods. Min cluster dim=%d. Wild bootstrap %s",
        unique_entities,
        n_treated,
        n_control,
        n_time,
        min_cluster_dim,
        "NEEDED" if needs_wild_bootstrap else "optional",
    )

    return result


if __name__ == "__main__":
    print("Wild cluster bootstrap module loaded")
    print("Functions:")
    print("  - wild_cluster_bootstrap()")
    print("  - compute_effective_clusters()")
