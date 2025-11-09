"""Tests for src/analysis/dimension_analysis.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.dimension_analysis import (
    DIMENSIONS,
    analyze_all_dimensions,
    analyze_single_dimension,
    compare_dimensions,
    compute_dimension_correlations,
    plot_dimension_comparison,
)
from src.utils.exceptions import DataValidationError


class TestAnalyzeSingleDimension:
    """Unit tests for analyze_single_dimension."""

    def test_analyze_stability_dimension(self, dimension_data: pd.DataFrame) -> None:
        result = analyze_single_dimension(dimension_data, dimension="S", n_deciles=8)

        assert result["dimension"] == "S"
        assert isinstance(result["summary"], pd.DataFrame)
        assert result["long_short"]["mean_ret"] > 0
        assert "p_value" in result["long_short"]

    def test_invalid_dimension_raises(self, dimension_data: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown dimension"):
            analyze_single_dimension(dimension_data, dimension="INVALID")

    def test_value_weighting_aligns_sign(self, dimension_data: pd.DataFrame) -> None:
        equal_result = analyze_single_dimension(dimension_data, dimension="R", weighting="equal")
        value_result = analyze_single_dimension(dimension_data, dimension="R", weighting="value")

        assert np.sign(equal_result["long_short"]["mean_ret"]) == np.sign(
            value_result["long_short"]["mean_ret"]
        )


class TestAnalyzeAllDimensions:
    """Tests for analyze_all_dimensions."""

    def test_all_dimensions_processed(self, dimension_data: pd.DataFrame) -> None:
        results = analyze_all_dimensions(dimension_data, n_deciles=6)
        assert set(results) == set(DIMENSIONS)

    def test_strong_dimensions_rank_high(self, dimension_data: pd.DataFrame) -> None:
        results = analyze_all_dimensions(dimension_data, n_deciles=10)
        comparison = compare_dimensions(results)

        assert comparison.iloc[0]["Dimension"] in {"S", "R"}
        assert bool(comparison.iloc[0]["Significant (p<0.05)"])


class TestCompareDimensions:
    """Tests for the comparison helper."""

    def test_comparison_has_expected_columns(self, dimension_data: pd.DataFrame) -> None:
        results = analyze_all_dimensions(dimension_data)
        comparison = compare_dimensions(results)

        expected_columns = {
            "Dimension",
            "Description",
            "Long-Short Return",
            "T-Statistic",
            "P-Value",
            "Significant (p<0.05)",
            "Ranking",
        }
        assert expected_columns.issubset(comparison.columns)
        assert comparison["Ranking"].is_monotonic_increasing


class TestComputeDimensionCorrelations:
    """Correlation helper tests."""

    def test_correlation_matrix_shape(self, dimension_data: pd.DataFrame) -> None:
        corr = compute_dimension_correlations(dimension_data)
        assert corr.shape == (len(DIMENSIONS), len(DIMENSIONS))
        assert np.allclose(np.diag(corr), 1.0)

    def test_no_dimensions_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_dimension_correlations(pd.DataFrame({"ticker": []}))


class TestPlotDimensionComparison:
    """Plotting helper tests."""

    def test_plot_outputs_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
        dimension_data: pd.DataFrame,
    ) -> None:
        results = analyze_all_dimensions(dimension_data, n_deciles=5)
        comparison = compare_dimensions(results)

        monkeypatch.setattr("src.analysis.dimension_analysis.plt.show", lambda: None)

        output_file = tmp_path / "dimensions.png"
        ax = plot_dimension_comparison(comparison, save_path=output_file)

        assert output_file.exists()
        assert "T-Statistic" in ax.get_xlabel()


class TestWorkflowGuards:
    """Integration tests for validation paths."""

    def test_analyze_all_dimensions_requires_data(self) -> None:
        with pytest.raises(DataValidationError):
            analyze_all_dimensions(pd.DataFrame({"ticker": []}))
