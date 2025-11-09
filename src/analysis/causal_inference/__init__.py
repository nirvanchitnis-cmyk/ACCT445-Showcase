"""
Causal Inference Modules

Includes:
- Difference-in-Differences estimation
- Parallel trends testing
"""

from src.analysis.causal_inference.difference_in_differences import (
    did_summary_table,
    prepare_did_data,
    run_did_regression,
)
from src.analysis.causal_inference.parallel_trends import (
    check_parallel_trends,
    placebo_test,
    plot_parallel_trends,
)

__all__ = [
    "prepare_did_data",
    "run_did_regression",
    "did_summary_table",
    "check_parallel_trends",
    "plot_parallel_trends",
    "placebo_test",
]
