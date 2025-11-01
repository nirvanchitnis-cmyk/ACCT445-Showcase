"""
Data Loader Utilities

Lightweight data loading for CNOI scores and market returns.
No large file dependencies - all data fetched on-demand.
"""

import pandas as pd
import yfinance as yf
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import warnings


def load_cnoi_data(filepath: str) -> pd.DataFrame:
    """
    Load CNOI scores from CSV.

    Args:
        filepath: Path to CNOI CSV file

    Returns:
        DataFrame with columns: cik, ticker, filing_date, CNOI, D, G, R, J, T, S, X, issuer

    Example:
        >>> cnoi = load_cnoi_data('config/sample_cnoi.csv')
        >>> cnoi[['ticker', 'CNOI', 'filing_date']].head()
    """
    df = pd.read_csv(filepath)

    # Ensure filing_date is datetime
    if 'filing_date' in df.columns:
        df['filing_date'] = pd.to_datetime(df['filing_date'])

    # Derive quarter from filing_date
    if 'filing_date' in df.columns and 'quarter' not in df.columns:
        df['quarter'] = df['filing_date'].dt.to_period('Q')

    print(f"Loaded {len(df)} CNOI records from {filepath}")
    print(f"  CIKs: {df['cik'].nunique()}")
    print(f"  Date range: {df['filing_date'].min()} to {df['filing_date'].max()}")
    print(f"  CNOI range: {df['CNOI'].min():.2f} to {df['CNOI'].max():.2f}")

    return df


def load_market_returns(
    tickers: List[str],
    start: str,
    end: str,
    frequency: str = 'daily'
) -> pd.DataFrame:
    """
    Download stock returns from Yahoo Finance.

    Args:
        tickers: List of ticker symbols
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        frequency: 'daily', 'weekly', 'monthly'

    Returns:
        DataFrame with columns: date, ticker, price, return, volume

    Example:
        >>> tickers = ['WFC', 'JPM', 'BAC']
        >>> returns = load_market_returns(tickers, '2023-01-01', '2025-11-01')
        >>> returns.head()
    """
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")

    all_data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start, end=end)

            if len(hist) == 0:
                warnings.warn(f"No data for {ticker}")
                continue

            # Calculate returns
            hist['return'] = hist['Close'].pct_change()
            hist['ticker'] = ticker
            hist = hist.reset_index()
            hist = hist.rename(columns={'Date': 'date', 'Close': 'price', 'Volume': 'volume'})

            all_data.append(hist[['date', 'ticker', 'price', 'return', 'volume']])

        except Exception as e:
            warnings.warn(f"Error downloading {ticker}: {e}")
            continue

    if len(all_data) == 0:
        raise ValueError("No data downloaded successfully")

    df = pd.concat(all_data, ignore_index=True)

    # Resample if needed
    if frequency == 'weekly':
        df = df.set_index('date').groupby('ticker').resample('W').agg({
            'price': 'last',
            'return': lambda x: (1 + x).prod() - 1,
            'volume': 'sum'
        }).reset_index()
    elif frequency == 'monthly':
        df = df.set_index('date').groupby('ticker').resample('M').agg({
            'price': 'last',
            'return': lambda x: (1 + x).prod() - 1,
            'volume': 'sum'
        }).reset_index()

    print(f"Downloaded {len(df)} observations for {df['ticker'].nunique()} tickers")

    return df


def compute_forward_returns(
    returns_df: pd.DataFrame,
    horizon: int = 1,
    frequency: str = 'quarterly'
) -> pd.DataFrame:
    """
    Compute forward returns for backtesting.

    Args:
        returns_df: DataFrame from load_market_returns()
        horizon: Number of periods forward (1 = next quarter)
        frequency: 'daily', 'weekly', 'monthly', 'quarterly'

    Returns:
        DataFrame with added 'ret_fwd' column

    Example:
        >>> returns = load_market_returns(['WFC'], '2023-01-01', '2025-11-01')
        >>> fwd = compute_forward_returns(returns, horizon=1, frequency='quarterly')
    """
    df = returns_df.copy()

    # Resample to target frequency if needed
    if frequency == 'quarterly':
        df['period'] = pd.PeriodIndex(df['date'], freq='Q')
        period_returns = df.groupby(['ticker', 'period'])['return'].apply(
            lambda x: (1 + x).prod() - 1
        ).reset_index()
        period_returns = period_returns.rename(columns={'return': 'ret_period'})

        # Compute forward returns
        period_returns['ret_fwd'] = period_returns.groupby('ticker')['ret_period'].shift(-horizon)

        df = df.merge(period_returns[['ticker', 'period', 'ret_fwd']], on=['ticker', 'period'], how='left')

    else:
        # For daily/weekly/monthly, simple shift
        df['ret_fwd'] = df.groupby('ticker')['return'].shift(-horizon)

    return df


def merge_cnoi_with_returns(
    cnoi_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    lag_days: int = 1
) -> pd.DataFrame:
    """
    Merge CNOI scores with forward returns.

    Enforces information timing: CNOI score from filing_date t
    predicts returns starting t + lag_days.

    Args:
        cnoi_df: DataFrame from load_cnoi_data()
        returns_df: DataFrame from load_market_returns()
        lag_days: Days after filing before signal is tradable

    Returns:
        Merged DataFrame with CNOI + forward returns

    Example:
        >>> cnoi = load_cnoi_data('config/sample_cnoi.csv')
        >>> returns = load_market_returns(cnoi['ticker'].unique(), '2023-01-01', '2025-11-01')
        >>> merged = merge_cnoi_with_returns(cnoi, returns, lag_days=1)
    """
    # Ensure tickers match
    common_tickers = set(cnoi_df['ticker'].dropna()) & set(returns_df['ticker'])
    print(f"Common tickers: {len(common_tickers)}")

    cnoi_sub = cnoi_df[cnoi_df['ticker'].isin(common_tickers)].copy()
    returns_sub = returns_df[returns_df['ticker'].isin(common_tickers)].copy()

    # Apply lag to filing_date
    cnoi_sub['decision_date'] = cnoi_sub['filing_date'] + timedelta(days=lag_days)

    # Merge on ticker + nearest date
    merged = pd.merge_asof(
        cnoi_sub.sort_values('decision_date'),
        returns_sub.sort_values('date'),
        left_on='decision_date',
        right_on='date',
        by='ticker',
        direction='forward',
        tolerance=pd.Timedelta('30 days')
    )

    print(f"Merged {len(merged)} observations")
    print(f"  Missing returns: {merged['return'].isna().sum()}")

    return merged


def create_sample_cnoi_file(
    full_cnoi_path: str,
    output_path: str,
    n_top: int = 20,
    n_bottom: int = 20
) -> None:
    """
    Create lightweight sample CNOI file (top + bottom banks only).

    Args:
        full_cnoi_path: Path to full CNOI CSV
        output_path: Where to save sample
        n_top: Number of most transparent banks
        n_bottom: Number of most opaque banks

    Example:
        >>> create_sample_cnoi_file(
        ...     '/path/to/cnoi_full.csv',
        ...     'config/sample_cnoi.csv',
        ...     n_top=20,
        ...     n_bottom=20
        ... )
    """
    df = pd.read_csv(full_cnoi_path)

    # Get latest CNOI per bank
    df['filing_date'] = pd.to_datetime(df['filing_date'])
    latest = df.sort_values('filing_date').groupby('cik').last().reset_index()

    # Top N (most transparent)
    top = latest.nsmallest(n_top, 'CNOI')

    # Bottom N (most opaque)
    bottom = latest.nlargest(n_bottom, 'CNOI')

    # Combine
    sample = pd.concat([top, bottom])

    # Get all historical filings for these banks
    sample_ciks = sample['cik'].unique()
    full_sample = df[df['cik'].isin(sample_ciks)]

    full_sample.to_csv(output_path, index=False)
    print(f"Created sample CNOI file: {output_path}")
    print(f"  Banks: {len(sample_ciks)}")
    print(f"  Filings: {len(full_sample)}")
    print(f"  File size: {len(full_sample) * 100 / len(df):.1f}% of original")


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Data Loader Demo")
    print("=" * 60)

    # Create sample file if full CNOI exists
    import os
    full_path = '../../../ACCT445-Banks/out/cnoi_top100_20251101120954.csv'
    if os.path.exists(full_path):
        create_sample_cnoi_file(
            full_path,
            'config/sample_cnoi.csv',
            n_top=20,
            n_bottom=20
        )
    else:
        print(f"Full CNOI file not found: {full_path}")
        print("Skipping sample creation")
