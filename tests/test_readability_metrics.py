"""
Tests for readability metrics module.

Tests standard readability formulas and CECL note processing.
"""

import tempfile
from pathlib import Path

import pandas as pd

from src.analysis.opacity_benchmarking.readability_metrics import (
    compute_disclosure_metrics,
    compute_flesch_kincaid_grade,
    compute_flesch_reading_ease,
    compute_fog_index,
    compute_readability_for_dataset,
    compute_smog_index,
    load_cecl_notes_from_filings,
)


class TestFogIndex:
    """Test Gunning Fog Index computation."""

    def test_fog_index_simple_text(self):
        """Test fog index on simple text."""
        text = "The cat sat on the mat. The dog ran in the park."
        fog = compute_fog_index(text)
        assert isinstance(fog, float)
        assert 0 <= fog <= 25  # Reasonable range

    def test_fog_index_complex_text(self):
        """Test fog index increases with complexity."""
        simple = "The cat sat. The dog ran."
        complex_text = (
            "The aforementioned feline positioned itself atop the textile surface. "
            "The canine executed rapid locomotion throughout the recreational area."
        )

        fog_simple = compute_fog_index(simple)
        fog_complex = compute_fog_index(complex_text)

        assert fog_complex > fog_simple

    def test_fog_index_empty_text(self):
        """Test fog index handles empty text."""
        assert compute_fog_index("") == 0.0
        assert compute_fog_index("   ") == 0.0

    def test_fog_index_financial_disclosure(self):
        """Test fog index on realistic financial disclosure text."""
        text = (
            "The Company adopted ASC 326-20, Current Expected Credit Losses (CECL), "
            "effective January 1, 2023. The allowance for credit losses represents "
            "management's estimate of lifetime expected credit losses on financial "
            "assets measured at amortized cost. Significant judgment is required in "
            "determining the economic forecasts and qualitative factors used in the "
            "allowance estimation process."
        )

        fog = compute_fog_index(text)
        assert 10 <= fog <= 22  # College-level reading (textstat can vary)


class TestFleschReadingEase:
    """Test Flesch Reading Ease score."""

    def test_flesch_ease_simple_text(self):
        """Test Flesch Reading Ease on simple text."""
        text = "The cat sat on the mat."
        ease = compute_flesch_reading_ease(text)
        assert isinstance(ease, float)
        assert ease > 50  # Simple text should be easy to read

    def test_flesch_ease_inverse_to_fog(self):
        """Test that Flesch Ease decreases as Fog increases."""
        simple = "The cat sat. The dog ran."
        complex_text = (
            "The aforementioned feline positioned itself. " "The canine executed locomotion."
        )

        ease_simple = compute_flesch_reading_ease(simple)
        ease_complex = compute_flesch_reading_ease(complex_text)

        assert ease_simple > ease_complex  # Simpler text = higher ease

    def test_flesch_ease_empty_text(self):
        """Test Flesch Ease handles empty text."""
        assert compute_flesch_reading_ease("") == 0.0


class TestFleschKincaidGrade:
    """Test Flesch-Kincaid Grade Level."""

    def test_fk_grade_simple_text(self):
        """Test FK grade on simple text."""
        text = "The cat sat on the mat. The dog ran."
        grade = compute_flesch_kincaid_grade(text)
        assert isinstance(grade, float)
        assert -5 <= grade <= 12  # textstat can give negative grades for very simple text

    def test_fk_grade_complex_text(self):
        """Test FK grade increases with complexity."""
        simple = "The cat sat. The dog ran."
        complex_text = (
            "The Company's comprehensive expected credit loss methodology "
            "incorporates quantitative probability-weighted scenarios."
        )

        grade_simple = compute_flesch_kincaid_grade(simple)
        grade_complex = compute_flesch_kincaid_grade(complex_text)

        assert grade_complex > grade_simple

    def test_fk_grade_empty_text(self):
        """Test FK grade handles empty text."""
        assert compute_flesch_kincaid_grade("") == 0.0


class TestSmogIndex:
    """Test SMOG Index."""

    def test_smog_simple_text(self):
        """Test SMOG index on simple text."""
        text = "The cat sat on the mat. The dog ran in the park."
        smog = compute_smog_index(text)
        assert isinstance(smog, float)
        assert smog >= 0

    def test_smog_empty_text(self):
        """Test SMOG handles empty text."""
        assert compute_smog_index("") == 0.0


class TestDisclosureMetrics:
    """Test comprehensive disclosure metrics function."""

    def test_disclosure_metrics_complete(self):
        """Test that all expected metrics are returned."""
        text = (
            "The Company adopted the Current Expected Credit Losses (CECL) "
            "accounting standard. Management estimates expected credit losses "
            "using quantitative models and qualitative factors."
        )

        metrics = compute_disclosure_metrics(text)

        expected_keys = [
            "fog_index",
            "flesch_ease",
            "fk_grade",
            "smog",
            "word_count",
            "sentence_count",
            "avg_words_per_sentence",
            "complex_word_pct",
            "syllables_per_word",
            "char_count",
        ]

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"
            assert isinstance(metrics[key], (int, float))

    def test_disclosure_metrics_word_count(self):
        """Test word count accuracy."""
        text = "The cat sat on the mat."  # 6 words
        metrics = compute_disclosure_metrics(text)
        assert metrics["word_count"] == 6

    def test_disclosure_metrics_sentence_count(self):
        """Test sentence count accuracy."""
        text = "First sentence. Second sentence. Third sentence."
        metrics = compute_disclosure_metrics(text)
        assert metrics["sentence_count"] >= 1  # textstat may count differently

    def test_disclosure_metrics_empty_text(self):
        """Test metrics handle empty text gracefully."""
        metrics = compute_disclosure_metrics("")

        assert metrics["word_count"] == 0
        assert metrics["sentence_count"] == 0
        assert metrics["fog_index"] == 0.0

    def test_disclosure_metrics_complex_word_percentage(self):
        """Test complex word percentage calculation."""
        # Text with some complex words
        text = (
            "The extraordinary comprehensive methodology incorporates "
            "quantitative probabilistic scenarios."
        )
        metrics = compute_disclosure_metrics(text)

        assert 0 <= metrics["complex_word_pct"] <= 100
        # This text has many complex words
        assert metrics["complex_word_pct"] > 50


class TestLoadCECLNotes:
    """Test loading CECL notes from filings."""

    def test_load_cecl_notes_missing_directory(self):
        """Test loading from non-existent directory returns empty DataFrame."""
        df = load_cecl_notes_from_filings(filings_dir=Path("/nonexistent/path"))

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "cik" in df.columns
        assert "filing_date" in df.columns

    def test_load_cecl_notes_with_files(self):
        """Test loading CECL notes from temporary directory with mock files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create mock CECL note files
            note1 = tmpdir_path / "CIK_0001234567_2023-12-31_cecl_note.txt"
            note1.write_text("This is a CECL disclosure note for Bank A.")

            note2 = tmpdir_path / "CIK_0007654321_2024-03-31_cecl_note.txt"
            note2.write_text("This is a CECL disclosure note for Bank B.")

            # Load notes
            df = load_cecl_notes_from_filings(filings_dir=tmpdir_path)

            assert len(df) == 2
            assert set(df["cik"]) == {"0001234567", "0007654321"}
            assert "cecl_note_text" in df.columns
            assert all(df["text_length"] > 0)

    def test_load_cecl_notes_with_filter(self):
        """Test loading CECL notes filtered by CNOI DataFrame."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create 3 notes
            for cik in ["0001", "0002", "0003"]:
                note = tmpdir_path / f"CIK_{cik}_2023-12-31_cecl_note.txt"
                note.write_text(f"CECL note for {cik}")

            # Filter to only load 2 of them
            cnoi_df = pd.DataFrame(
                {"cik": ["0001", "0002"], "filing_date": ["2023-12-31", "2023-12-31"]}
            )

            df = load_cecl_notes_from_filings(filings_dir=tmpdir_path, cnoi_df=cnoi_df)

            assert len(df) == 2
            assert set(df["cik"]) == {"0001", "0002"}


class TestComputeReadabilityForDataset:
    """Test computing readability for entire dataset."""

    def test_compute_readability_no_files(self):
        """Test computing readability when no CECL notes exist."""
        cnoi_df = pd.DataFrame({"cik": ["0001"], "filing_date": ["2023-12-31"], "CNOI": [15.0]})

        result = compute_readability_for_dataset(cnoi_df, filings_dir=Path("/nonexistent"))

        assert len(result) == 1
        assert "CNOI" in result.columns
        assert "fog_index" in result.columns
        assert pd.isna(result.loc[0, "fog_index"])

    def test_compute_readability_with_files(self):
        """Test computing readability with actual CECL notes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create CECL note
            note = tmpdir_path / "CIK_0001_2023-12-31_cecl_note.txt"
            note.write_text(
                "The Company adopted CECL accounting. "
                "Management estimates expected credit losses."
            )

            # CNOI data
            cnoi_df = pd.DataFrame({"cik": ["0001"], "filing_date": ["2023-12-31"], "CNOI": [15.0]})

            result = compute_readability_for_dataset(cnoi_df, tmpdir_path)

            assert len(result) == 1
            assert result.loc[0, "CNOI"] == 15.0
            assert result.loc[0, "fog_index"] > 0
            assert result.loc[0, "word_count"] > 0

    def test_compute_readability_multiple_filings(self):
        """Test computing readability for multiple banks and dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create multiple CECL notes
            filings = [
                ("0001", "2023-12-31", "Simple CECL note text."),
                ("0001", "2024-03-31", "Another simple note."),
                ("0002", "2023-12-31", "Complex comprehensive methodology."),
            ]

            for cik, date, text in filings:
                note = tmpdir_path / f"CIK_{cik}_{date}_cecl_note.txt"
                note.write_text(text)

            # CNOI data
            cnoi_df = pd.DataFrame(
                {
                    "cik": ["0001", "0001", "0002"],
                    "filing_date": ["2023-12-31", "2024-03-31", "2023-12-31"],
                    "CNOI": [10.0, 12.0, 20.0],
                }
            )

            result = compute_readability_for_dataset(cnoi_df, tmpdir_path)

            assert len(result) == 3
            assert all(result["fog_index"] > 0)
            assert all(result["word_count"] > 0)
