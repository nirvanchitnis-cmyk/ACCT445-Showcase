"""
CIK to Ticker Mapper

Uses SEC's official company tickers JSON API to map CIKs to stock tickers.
No manual lookup required - all automated and reproducible.

References:
- SEC Company Tickers API: https://www.sec.gov/files/company_tickers.json
- CIK format: 10-digit zero-padded strings
"""

import time
from pathlib import Path

import pandas as pd

import src.data.sec_api_client as sec_api_client
from src.utils.exceptions import DataValidationError, ExternalAPIError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OVERRIDE_PATH = PROJECT_ROOT / "config" / "cik_ticker_overrides.csv"


def fetch_sec_ticker_mapping(cache_path: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch official SEC CIK → Ticker mapping from SEC API.

    Args:
        cache_path: Optional path to cache the mapping CSV
        use_cache: Whether to honor the SEC client cache (24h TTL)

    Returns:
        DataFrame with columns: cik, ticker, title (company name), exchange

    Example:
        >>> mapping = fetch_sec_ticker_mapping()
        >>> mapping[mapping['ticker'] == 'WFC']
           cik  ticker                   title exchange
        0  72971   WFC  WELLS FARGO & COMPANY   NYSE
    """
    logger.info("Fetching SEC ticker mapping via cached client")
    try:
        mapping = sec_api_client.fetch_sec_ticker_mapping(use_cache=use_cache)
    except ExternalAPIError:
        raise
    except Exception as exc:  # pragma: no cover - defensive guard
        raise ExternalAPIError("Unexpected error fetching SEC mapping") from exc

    records = [
        {"cik": int(cik), "ticker": payload["ticker"], "title": payload["title"]}
        for cik, payload in mapping.items()
    ]
    df = pd.DataFrame.from_records(records, columns=["cik", "ticker", "title"])

    logger.info("Fetched %s company ticker mappings", len(df))

    # Cache if requested
    if cache_path:
        df.to_csv(cache_path, index=False)
        logger.info("Cached mapping to %s", cache_path)

    return df


def load_override_mapping(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load manual CIK → ticker overrides when SEC data is incomplete.

    Args:
        path: Optional path to override CSV (defaults to config/cik_ticker_overrides.csv)

    Returns:
        DataFrame with columns: cik, ticker, company_name (may be empty)
    """

    target = Path(path) if path is not None else OVERRIDE_PATH
    if not target.exists():
        logger.debug("No override mapping found at %s", target)
        return pd.DataFrame(columns=["cik", "ticker", "company_name"])

    overrides = pd.read_csv(target)
    required_cols = {"cik", "ticker"}
    if not required_cols.issubset(overrides.columns):
        raise DataValidationError(
            f"Override mapping at {target} missing columns: {sorted(required_cols - set(overrides.columns))}"
        )

    overrides = overrides.copy()
    overrides["cik"] = overrides["cik"].astype(int)
    if "company_name" not in overrides.columns:
        overrides["company_name"] = None

    logger.info("Loaded %s manual CIK overrides from %s", len(overrides), target)
    return overrides


def map_cik_to_ticker(cik: int, mapping_df: pd.DataFrame) -> str | None:
    """
    Map single CIK to ticker.

    Args:
        cik: CIK number (integer)
        mapping_df: DataFrame from fetch_sec_ticker_mapping()

    Returns:
        Ticker symbol (str) or None if not found

    Example:
        >>> mapping = fetch_sec_ticker_mapping()
        >>> map_cik_to_ticker(72971, mapping)
        'WFC'
    """
    if not {"cik", "ticker"}.issubset(mapping_df.columns):
        raise DataValidationError("mapping_df must contain 'cik' and 'ticker' columns")

    result = mapping_df[mapping_df["cik"] == cik]
    if len(result) == 0:
        return None
    return result.iloc[0]["ticker"]


def enrich_cnoi_with_tickers(
    cnoi_df: pd.DataFrame, cik_col: str = "cik", mapping_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Add ticker column to CNOI dataframe.

    Args:
        cnoi_df: DataFrame with CIK column
        cik_col: Name of CIK column (default: 'cik')
        mapping_df: Optional pre-fetched mapping (fetches if None)

    Returns:
        DataFrame with added 'ticker' and 'company_name' columns

    Example:
        >>> cnoi = pd.read_csv('cnoi_scores.csv')
        >>> cnoi_enriched = enrich_cnoi_with_tickers(cnoi)
        >>> cnoi_enriched[['cik', 'ticker', 'CNOI']].head()
    """
    if cik_col not in cnoi_df.columns:
        raise DataValidationError(f"{cik_col} column missing from CNOI dataframe")

    if mapping_df is None:
        mapping_df = fetch_sec_ticker_mapping()

    # Ensure CIK is integer
    cnoi_df = cnoi_df.copy()
    cnoi_df[cik_col] = cnoi_df[cik_col].astype(int)

    overrides = load_override_mapping()

    # Merge
    enriched = cnoi_df.merge(
        mapping_df[["cik", "ticker", "title"]], left_on=cik_col, right_on="cik", how="left"
    )

    enriched = enriched.rename(columns={"title": "company_name"})

    if not overrides.empty:
        overrides = overrides.rename(
            columns={"ticker": "override_ticker", "company_name": "override_company_name"}
        )
        enriched = enriched.merge(overrides, on="cik", how="left")
        missing_before = int(enriched["ticker"].isna().sum())
        enriched["ticker"] = enriched["ticker"].combine_first(enriched["override_ticker"])
        if "company_name" in enriched.columns:
            enriched["company_name"] = enriched["company_name"].combine_first(
                enriched["override_company_name"]
            )
        else:
            enriched["company_name"] = enriched["override_company_name"]
        overrides_used = missing_before - int(enriched["ticker"].isna().sum())
        enriched = enriched.drop(columns=["override_ticker", "override_company_name"])
        if overrides_used > 0:
            logger.info("Applied %s manual ticker overrides.", overrides_used)

    # Report missing tickers
    missing = enriched["ticker"].isna().sum()
    if missing > 0:
        missing_ciks = enriched[enriched["ticker"].isna()][cik_col].unique()
        logger.warning("Missing %s tickers in SEC mapping: %s", missing, missing_ciks)

    return enriched


def get_ticker_batch(ciks: list, rate_limit_delay: float = 0.1) -> dict[int, str]:
    """
    Batch fetch tickers for multiple CIKs with rate limiting.

    Args:
        ciks: List of CIK integers
        rate_limit_delay: Delay between requests (SEC limit: 10 req/sec)

    Returns:
        Dict mapping CIK → Ticker

    Example:
        >>> ciks = [72971, 19617, 70858]  # WFC, JPM, BAC
        >>> tickers = get_ticker_batch(ciks)
        >>> tickers
        {72971: 'WFC', 19617: 'JPM', 70858: 'BAC'}
    """
    # Fetch full mapping once (more efficient than individual requests)
    mapping_df = fetch_sec_ticker_mapping()
    overrides = load_override_mapping()
    override_lookup = (
        dict(zip(overrides["cik"], overrides["ticker"])) if not overrides.empty else {}
    )

    result = {}
    for cik in ciks:
        ticker = map_cik_to_ticker(cik, mapping_df)
        if not ticker:
            ticker = override_lookup.get(cik)
            if ticker:
                logger.debug("Using manual override for CIK %s -> %s", cik, ticker)
        if ticker:
            result[cik] = ticker
        else:
            logger.warning("Ticker not found for CIK %s", cik)
        time.sleep(rate_limit_delay)  # Respectful rate limiting

    return result


def _demo() -> None:  # pragma: no cover - convenience execution path
    logger.info("=" * 60)
    logger.info("SEC CIK → Ticker Mapping Demo")
    logger.info("=" * 60)

    # Fetch mapping
    try:
        mapping = fetch_sec_ticker_mapping(cache_path="config/sec_ticker_mapping.csv")
    except ExternalAPIError as exc:  # pragma: no cover - network failures
        logger.error("Failed to fetch SEC mapping: %s", exc)
        return

    # Test known banks
    test_ciks = {
        72971: "Wells Fargo",
        19617: "JPMorgan Chase",
        70858: "Bank of America",
        707605: "AmeriServ Financial",
        903419: "Alerus Financial",
    }

    logger.info("Test CIK → Ticker Mapping:")
    for cik, name in test_ciks.items():
        ticker = map_cik_to_ticker(cik, mapping)
        logger.info("CIK %s -> %s (%s)", cik, ticker, name)

    logger.info("✓ Mapping successful!")


if __name__ == "__main__":  # pragma: no cover
    _demo()
