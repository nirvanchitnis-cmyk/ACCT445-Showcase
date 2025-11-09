"""
Opacity benchmarking and validation module.

This module validates the CNOI (CECL Note Opacity Index) construct
by benchmarking against established disclosure quality measures.

Modules:
    readability_metrics: Standard readability and disclosure quality metrics
    opacity_validation: CNOI construct validation against external benchmarks
"""

from .opacity_validation import (
    compute_cnoi_readability_correlations,
    dimension_contribution_analysis,
    horse_race_regression,
)
from .readability_metrics import (
    compute_disclosure_metrics,
    compute_flesch_kincaid_grade,
    compute_flesch_reading_ease,
    compute_fog_index,
    compute_smog_index,
)

__all__ = [
    "compute_fog_index",
    "compute_flesch_reading_ease",
    "compute_flesch_kincaid_grade",
    "compute_smog_index",
    "compute_disclosure_metrics",
    "compute_cnoi_readability_correlations",
    "dimension_contribution_analysis",
    "horse_race_regression",
]
