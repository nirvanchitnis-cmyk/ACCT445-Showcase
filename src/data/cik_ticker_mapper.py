"""
CIK to Ticker Mapper

Uses SEC's official company tickers JSON API to map CIKs to stock tickers.
No manual lookup required - all automated and reproducible.

References:
- SEC Company Tickers API: https://www.sec.gov/files/company_tickers.json
- CIK format: 10-digit zero-padded strings
"""

import requests
import pandas as pd
from typing import Dict, Optional
import time


def fetch_sec_ticker_mapping(cache_path: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch official SEC CIK → Ticker mapping from SEC API.

    Args:
        cache_path: Optional path to cache the mapping CSV

    Returns:
        DataFrame with columns: cik, ticker, title (company name), exchange

    Example:
        >>> mapping = fetch_sec_ticker_mapping()
        >>> mapping[mapping['ticker'] == 'WFC']
           cik  ticker                   title exchange
        0  72971   WFC  WELLS FARGO & COMPANY   NYSE
    """
    url = "https://www.sec.gov/files/company_tickers.json"

    # SEC requires User-Agent header
    headers = {
        'User-Agent': 'ACCT445 Research Project (academic use)',
        'Accept-Encoding': 'gzip, deflate'
    }

    print(f"Fetching SEC ticker mapping from {url}...")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    # API returns dict with numeric keys
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            'cik': int(v['cik_str']),
            'ticker': v['ticker'],
            'title': v['title']
        }
        for v in data.values()
    ])

    print(f"Fetched {len(df):,} company ticker mappings")

    # Cache if requested
    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"Cached mapping to {cache_path}")

    return df


def map_cik_to_ticker(cik: int, mapping_df: pd.DataFrame) -> Optional[str]:
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
    result = mapping_df[mapping_df['cik'] == cik]
    if len(result) == 0:
        return None
    return result.iloc[0]['ticker']


def enrich_cnoi_with_tickers(
    cnoi_df: pd.DataFrame,
    cik_col: str = 'cik',
    mapping_df: Optional[pd.DataFrame] = None
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
    if mapping_df is None:
        mapping_df = fetch_sec_ticker_mapping()

    # Ensure CIK is integer
    cnoi_df = cnoi_df.copy()
    cnoi_df[cik_col] = cnoi_df[cik_col].astype(int)

    # Merge
    enriched = cnoi_df.merge(
        mapping_df[['cik', 'ticker', 'title']],
        left_on=cik_col,
        right_on='cik',
        how='left'
    )

    enriched = enriched.rename(columns={'title': 'company_name'})

    # Report missing tickers
    missing = enriched['ticker'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} CIKs not found in SEC mapping")
        print("Missing CIKs:", enriched[enriched['ticker'].isna()][cik_col].unique())

    return enriched


def get_ticker_batch(ciks: list, rate_limit_delay: float = 0.1) -> Dict[int, str]:
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

    result = {}
    for cik in ciks:
        ticker = map_cik_to_ticker(cik, mapping_df)
        if ticker:
            result[cik] = ticker
        time.sleep(rate_limit_delay)  # Respectful rate limiting

    return result


if __name__ == "__main__":
    # Demo usage
    print("=" * 60)
    print("SEC CIK → Ticker Mapping Demo")
    print("=" * 60)

    # Fetch mapping
    mapping = fetch_sec_ticker_mapping(cache_path='config/sec_ticker_mapping.csv')

    # Test known banks
    test_ciks = {
        72971: 'Wells Fargo',
        19617: 'JPMorgan Chase',
        70858: 'Bank of America',
        707605: 'AmeriServ Financial',
        903419: 'Alerus Financial'
    }

    print("\nTest CIK → Ticker Mapping:")
    print("-" * 60)
    for cik, name in test_ciks.items():
        ticker = map_cik_to_ticker(cik, mapping)
        print(f"CIK {cik:>8} → {ticker:>6}  ({name})")

    print("\n" + "=" * 60)
    print("✓ Mapping successful!")
