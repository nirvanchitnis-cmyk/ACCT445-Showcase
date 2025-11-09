"""Factor models for risk-adjusted performance analysis."""

from src.analysis.factor_models.alpha_decomposition import (
    alpha_attribution,
    carhart_alpha,
    jensen_alpha,
)
from src.analysis.factor_models.alpha_robust import (
    compute_alpha_with_dsr,
    deflated_sharpe_ratio,
    rolling_alpha,
)
from src.analysis.factor_models.fama_french import (
    compute_abnormal_return,
    compute_expected_return,
    estimate_factor_loadings,
)

__all__ = [
    "estimate_factor_loadings",
    "compute_expected_return",
    "compute_abnormal_return",
    "jensen_alpha",
    "carhart_alpha",
    "alpha_attribution",
    "deflated_sharpe_ratio",
    "rolling_alpha",
    "compute_alpha_with_dsr",
]
