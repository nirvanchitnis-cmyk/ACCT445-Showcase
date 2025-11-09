"""
Standard readability and disclosure quality metrics.

This module computes established readability metrics to benchmark
against the CNOI index. These metrics have been validated in prior
finance and accounting research.

References:
    - Flesch (1948) - Reading ease formula
    - Kincaid et al. (1975) - Grade level formula
    - Gunning (1952) - Fog Index for readability
    - Li (2008) - Disclosure length and obfuscation (JAR)
    - Loughran & McDonald (2014) - Financial text complexity
"""

from pathlib import Path

import pandas as pd
import textstat


def compute_fog_index(text: str) -> float:
    """
    Compute Gunning Fog Index.

    Formula: 0.4 × [(words/sentences) + 100 × (complex_words/words)]

    Higher values indicate harder to read text (more opaque).

    Args:
        text: Raw text from CECL note

    Returns:
        Fog index (typically 6-17, where >12 = college level)

    Examples:
        >>> text = "The bank disclosed credit losses. Provisions increased."
        >>> fog = compute_fog_index(text)
        >>> assert 6 <= fog <= 17

    References:
        Gunning, R. (1952). The Technique of Clear Writing.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    return textstat.gunning_fog(text)


def compute_flesch_reading_ease(text: str) -> float:
    """
    Compute Flesch Reading Ease score.

    Formula: 206.835 - 1.015×(words/sentences) - 84.6×(syllables/words)

    Lower values indicate harder to read text (more opaque).
    Range: 0-100, where:
        - 90-100: Very easy (5th grade)
        - 60-70: Standard (8th-9th grade)
        - 30-50: Difficult (college)
        - 0-30: Very difficult (college graduate)

    Args:
        text: Raw text from CECL note

    Returns:
        Flesch Reading Ease score (0-100 scale)

    Examples:
        >>> easy_text = "The cat sat on the mat."
        >>> hard_text = "The aforementioned feline positioned itself atop the textile surface."
        >>> compute_flesch_reading_ease(easy_text) > compute_flesch_reading_ease(hard_text)
        True

    References:
        Flesch, R. (1948). A new readability yardstick. Journal of Applied Psychology.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    return textstat.flesch_reading_ease(text)


def compute_flesch_kincaid_grade(text: str) -> float:
    """
    Compute Flesch-Kincaid Grade Level.

    Estimates U.S. school grade level needed to understand text.
    Higher values indicate more opaque text.

    Args:
        text: Raw text from CECL note

    Returns:
        Grade level (e.g., 12.0 = 12th grade, 16.0 = college senior)

    Examples:
        >>> text = "The expected credit loss model requires significant judgment."
        >>> grade = compute_flesch_kincaid_grade(text)
        >>> assert 8 <= grade <= 18  # Typical range for financial disclosures

    References:
        Kincaid, J. P., et al. (1975). Derivation of new readability formulas.
        Naval Technical Training Command Research Branch Report.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    return textstat.flesch_kincaid_grade(text)


def compute_smog_index(text: str) -> float:
    """
    Compute SMOG (Simple Measure of Gobbledygook) Index.

    Estimates years of education needed to understand text.
    Focuses on polysyllabic words (3+ syllables).

    Args:
        text: Raw text from CECL note

    Returns:
        SMOG index (grade level equivalent)

    References:
        McLaughlin, G. H. (1969). SMOG grading: A new readability formula.
        Journal of Reading, 12(8), 639-646.
    """
    if not text or len(text.strip()) == 0:
        return 0.0

    return textstat.smog_index(text)


def compute_disclosure_metrics(text: str) -> dict[str, float]:
    """
    Compute comprehensive disclosure quality metrics.

    This function computes multiple readability and complexity metrics
    in one pass for efficiency.

    Args:
        text: Raw text from CECL note

    Returns:
        Dictionary with the following keys:
            - fog_index: Gunning Fog Index (higher = more opaque)
            - flesch_ease: Flesch Reading Ease (lower = more opaque)
            - fk_grade: Flesch-Kincaid Grade Level (higher = more opaque)
            - smog: SMOG Index (higher = more opaque)
            - word_count: Total words
            - sentence_count: Total sentences
            - avg_words_per_sentence: Mean sentence length
            - complex_word_pct: Percentage of words >2 syllables
            - syllables_per_word: Average syllables per word
            - char_count: Total characters

    Examples:
        >>> text = "The bank disclosed expected credit losses. Provisions increased significantly."
        >>> metrics = compute_disclosure_metrics(text)
        >>> assert 'fog_index' in metrics
        >>> assert 'flesch_ease' in metrics
        >>> assert metrics['word_count'] > 0

    Notes:
        All metrics are computed using the textstat library, which
        implements standard readability formulas from the linguistics
        and education literature.
    """
    if not text or len(text.strip()) == 0:
        return {
            "fog_index": 0.0,
            "flesch_ease": 0.0,
            "fk_grade": 0.0,
            "smog": 0.0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_words_per_sentence": 0.0,
            "complex_word_pct": 0.0,
            "syllables_per_word": 0.0,
            "char_count": 0,
        }

    # Core readability metrics
    fog = compute_fog_index(text)
    flesch_ease = compute_flesch_reading_ease(text)
    fk_grade = compute_flesch_kincaid_grade(text)
    smog = compute_smog_index(text)

    # Basic text statistics
    word_count = textstat.lexicon_count(text, removepunct=True)
    sentence_count = textstat.sentence_count(text)
    avg_words = textstat.avg_sentence_length(text) if sentence_count > 0 else 0.0

    # Complexity metrics
    difficult_words = textstat.difficult_words(text)
    complex_pct = (difficult_words / word_count * 100) if word_count > 0 else 0.0

    syllable_count = textstat.syllable_count(text)
    syllables_per_word = (syllable_count / word_count) if word_count > 0 else 0.0

    char_count = textstat.char_count(text, ignore_spaces=False)

    return {
        "fog_index": fog,
        "flesch_ease": flesch_ease,
        "fk_grade": fk_grade,
        "smog": smog,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": avg_words,
        "complex_word_pct": complex_pct,
        "syllables_per_word": syllables_per_word,
        "char_count": char_count,
    }


def load_cecl_notes_from_filings(
    filings_dir: Path | None = None, cnoi_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Load raw CECL note text for each bank-filing.

    This function expects CECL notes to be stored as separate text files
    following the naming convention:
        CIK_{cik}_{filing_date}_cecl_note.txt

    Args:
        filings_dir: Directory containing CECL note text files
                    Default: data/sec_filings
        cnoi_df: Optional DataFrame with CIK and filing_date columns
                to filter which notes to load

    Returns:
        DataFrame with columns:
            - cik: SEC Central Index Key
            - filing_date: Filing date (YYYY-MM-DD)
            - cecl_note_text: Raw CECL note text
            - text_length: Number of characters

    Examples:
        >>> # Load all CECL notes from default directory
        >>> notes_df = load_cecl_notes_from_filings()

        >>> # Load only notes for specific filings
        >>> cnoi_df = pd.DataFrame({
        ...     'cik': ['0001234567', '0001234568'],
        ...     'filing_date': ['2023-12-31', '2023-12-31']
        ... })
        >>> notes_df = load_cecl_notes_from_filings(cnoi_df=cnoi_df)

    Notes:
        If the text files don't exist, this function will return an empty
        DataFrame. For testing purposes, you can generate mock CECL notes
        or use pre-extracted samples.
    """
    if filings_dir is None:
        filings_dir = Path("data/sec_filings")

    filings_dir = Path(filings_dir)

    if not filings_dir.exists():
        # Return empty DataFrame if directory doesn't exist
        return pd.DataFrame(columns=["cik", "filing_date", "cecl_note_text", "text_length"])

    rows = []

    # Find all CECL note text files
    for file_path in filings_dir.glob("*_cecl_note.txt"):
        try:
            # Parse filename: CIK_{cik}_{date}_cecl_note.txt
            filename = file_path.stem  # Remove .txt
            parts = filename.split("_")

            if len(parts) >= 4 and parts[0] == "CIK":
                cik = parts[1]
                filing_date = parts[2]

                # Load text
                text = file_path.read_text(encoding="utf-8")

                # Filter by CNOI DataFrame if provided
                if cnoi_df is not None:
                    matching = cnoi_df[
                        (cnoi_df["cik"] == cik) & (cnoi_df["filing_date"] == filing_date)
                    ]
                    if len(matching) == 0:
                        continue

                rows.append(
                    {
                        "cik": cik,
                        "filing_date": filing_date,
                        "cecl_note_text": text,
                        "text_length": len(text),
                    }
                )
        except Exception:
            # Skip files that can't be parsed
            continue

    return pd.DataFrame(rows)


def compute_readability_for_dataset(
    cnoi_df: pd.DataFrame, filings_dir: Path | None = None
) -> pd.DataFrame:
    """
    Compute readability metrics for entire CNOI dataset.

    This is a convenience function that:
    1. Loads CECL note text for all filings in CNOI dataset
    2. Computes readability metrics for each note
    3. Merges with CNOI scores

    Args:
        cnoi_df: DataFrame with CNOI scores (must have 'cik' and 'filing_date')
        filings_dir: Directory containing CECL note text files

    Returns:
        DataFrame with CNOI scores and readability metrics merged

    Examples:
        >>> cnoi_df = pd.DataFrame({
        ...     'cik': ['0001234567'],
        ...     'filing_date': ['2023-12-31'],
        ...     'CNOI': [15.2]
        ... })
        >>> results_df = compute_readability_for_dataset(cnoi_df)
        >>> assert 'fog_index' in results_df.columns
        >>> assert 'CNOI' in results_df.columns
    """
    # Load CECL notes
    notes_df = load_cecl_notes_from_filings(filings_dir, cnoi_df)

    if len(notes_df) == 0:
        # Return CNOI df with empty readability columns
        result = cnoi_df.copy()
        for col in [
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
        ]:
            result[col] = None
        return result

    # Compute metrics for each note
    metrics_list = []
    for _, row in notes_df.iterrows():
        metrics = compute_disclosure_metrics(row["cecl_note_text"])
        metrics["cik"] = row["cik"]
        metrics["filing_date"] = row["filing_date"]
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    # Merge with CNOI scores
    result = cnoi_df.merge(metrics_df, on=["cik", "filing_date"], how="left")

    return result
