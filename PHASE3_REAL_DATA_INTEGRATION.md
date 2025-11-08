# Phase 3: Real Data Integration & Validation

**Phase**: 3 of 5
**Estimated Time**: 25-35 hours
**Dependencies**: Phase 2 complete (all analysis modules ready)
**Status**: 🔴 Blocked (requires Phase 2)

---

## 🎯 Objectives

Create end-to-end workflows with real yfinance market data:
1. Build 4 comprehensive Jupyter notebooks (data exploration, decile, event study, panel regression)
2. Implement robust yfinance integration (rate limiting, retries, caching, error handling)
3. Generate real results for all 40 sample banks
4. Validate methodology against academic benchmarks
5. Document limitations and performance analysis

**Success Criteria**:
- ✅ All 4 notebooks run end-to-end without errors
- ✅ Real market data fetched successfully from yfinance
- ✅ Results tables generated and saved to `results/`
- ✅ Methodology validated (decile spread significant, reasonable magnitudes)
- ✅ Rate limiting prevents API bans
- ✅ Notebooks have professional markdown explanations

---

## 📋 Task Breakdown

### Task 3.1: Enhance yfinance Integration (5-6 hours)

#### 3.1.1: Create `src/utils/market_data.py`

**Purpose**: Robust yfinance wrapper with caching, rate limiting, retries

```python
"""
Robust market data fetching with yfinance.

Features:
- Automatic retry with exponential backoff
- Rate limiting to avoid API bans
- Disk caching of downloaded data
- Progress tracking for bulk downloads
- Validation of data quality
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import time
from functools import wraps
import pickle
from src.utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def rate_limited(calls_per_second: float = 2.0):
    """Decorator to rate limit function calls."""
    min_interval = 1.0 / calls_per_second

    def decorator(func):
        last_called = [0.0]

        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            wait_time = min_interval - elapsed

            if wait_time > 0:
                time.sleep(wait_time)

            result = func(*args, **kwargs)
            last_called[0] = time.time()
            return result

        return wrapper

    return decorator


@rate_limited(calls_per_second=2.0)
def fetch_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    use_cache: bool = True,
    max_retries: int = 3
) -> pd.DataFrame:
    """
    Fetch historical data for single ticker with retry logic.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        use_cache: Use cached data if available
        max_retries: Maximum retry attempts

    Returns:
        DataFrame with columns: date, ticker, close, ret

    Example:
        >>> df = fetch_ticker_data("AAPL", "2023-01-01", "2023-12-31")
    """
    cache_file = CACHE_DIR / f"{ticker}_{start_date}_{end_date}.pkl"

    # Check cache
    if use_cache and cache_file.exists():
        logger.debug(f"Loading {ticker} from cache")
        return pd.read_pickle(cache_file)

    # Fetch from yfinance with retries
    for attempt in range(max_retries):
        try:
            logger.info(f"Fetching {ticker} (attempt {attempt + 1}/{max_retries})")

            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Compute returns
            df = pd.DataFrame({
                "date": data.index,
                "ticker": ticker,
                "close": data["Close"].values,
            })
            df["ret"] = df["close"].pct_change()

            # Validate
            if df["ret"].abs().max() > 0.5:  # 50% daily return is suspicious
                logger.warning(f"{ticker}: Extreme returns detected, check data quality")

            # Cache
            df.to_pickle(cache_file)

            logger.info(f"✓ {ticker}: {len(df)} days")
            return df

        except Exception as e:
            logger.warning(f"Error fetching {ticker} (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait}s...")
                time.sleep(wait)

    logger.error(f"Failed to fetch {ticker} after {max_retries} attempts")
    return pd.DataFrame()


def fetch_bulk_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch data for multiple tickers with progress tracking.

    Args:
        tickers: List of ticker symbols
        start_date: Start date
        end_date: End date
        use_cache: Use cached data

    Returns:
        Combined DataFrame

    Example:
        >>> tickers = ["AAPL", "MSFT", "GOOGL"]
        >>> df = fetch_bulk_data(tickers, "2023-01-01", "2023-12-31")
    """
    all_data = []
    failed = []

    logger.info(f"Fetching {len(tickers)} tickers from {start_date} to {end_date}")

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"Progress: {i}/{len(tickers)} ({i/len(tickers):.1%})")

        df = fetch_ticker_data(ticker, start_date, end_date, use_cache=use_cache)

        if not df.empty:
            all_data.append(df)
        else:
            failed.append(ticker)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"✓ Successfully fetched {len(all_data)} tickers ({len(combined)} total rows)")
    else:
        combined = pd.DataFrame()
        logger.warning("No data fetched for any ticker")

    if failed:
        logger.warning(f"Failed tickers ({len(failed)}): {failed}")

    return combined


def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    Validate fetched data quality.

    Args:
        df: Market data DataFrame

    Returns:
        Dictionary with quality metrics

    Example:
        >>> quality = validate_data_quality(market_df)
        >>> print(f"Coverage: {quality['coverage']:.1%}")
    """
    total_rows = len(df)
    missing_returns = df["ret"].isna().sum()
    extreme_returns = (df["ret"].abs() > 0.3).sum()  # >30% daily
    negative_prices = (df["close"] <= 0).sum()

    unique_tickers = df["ticker"].nunique()
    avg_days_per_ticker = df.groupby("ticker").size().mean()

    quality = {
        "total_rows": total_rows,
        "unique_tickers": unique_tickers,
        "avg_days_per_ticker": avg_days_per_ticker,
        "missing_returns": missing_returns,
        "extreme_returns": extreme_returns,
        "negative_prices": negative_prices,
        "coverage": 1 - (missing_returns / total_rows) if total_rows > 0 else 0,
    }

    logger.info(f"Data quality: {unique_tickers} tickers, {quality['coverage']:.1%} coverage")

    return quality


if __name__ == "__main__":
    # Demo
    tickers = ["BAC", "JPM", "WFC", "C", "USB"]
    df = fetch_bulk_data(tickers, "2023-01-01", "2023-12-31")

    quality = validate_data_quality(df)
    print("\n=== DATA QUALITY ===")
    for key, value in quality.items():
        print(f"{key}: {value}")
```

**Tests**: `tests/test_market_data.py` (mock yfinance calls)

**Checkpoint 3.1**: Robust yfinance integration complete

---

### Task 3.2: Notebook 1 - Data Exploration (6-7 hours)

#### 3.2.1: Create `notebooks/01_data_exploration.ipynb`

**Structure**:
1. Introduction & Research Question
2. Load CNOI Data (sample_cnoi.csv)
3. Enrich with Tickers (CIK → ticker mapping)
4. Exploratory Data Analysis:
   - CNOI distribution (histogram, summary stats)
   - Dimension correlations (heatmap)
   - Top 10 most transparent banks
   - Top 10 most opaque banks
   - CNOI trends over time
5. Fetch Market Data (yfinance for all tickers)
6. Data Quality Assessment
7. Merge CNOI with Returns
8. Initial Observations

**Key Cells**:

```python
# Cell 1: Introduction
"""
# ACCT445 Data Exploration: Bank Disclosure Opacity & Stock Returns

**Research Question**: Do banks with opaque CECL disclosures (high CNOI) underperform?

**Data**:
- CNOI scores from 40 banks (top 20 + bottom 20)
- Stock returns from Yahoo Finance
- Period: 2023-2024

**Hypothesis**: High CNOI → Lower returns, higher volatility
"""

# Cell 2: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.cik_ticker_mapper import enrich_cnoi_with_tickers
from src.utils.data_loader import load_cnoi_data
from src.utils.market_data import fetch_bulk_data, validate_data_quality
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_style("whitegrid")

# Cell 3: Load CNOI data
cnoi_df = load_cnoi_data("../config/sample_cnoi.csv")
print(f"Loaded {len(cnoi_df)} CNOI filings from {cnoi_df['cik'].nunique()} banks")
cnoi_df.head()

# Cell 4: Summary statistics
cnoi_df["CNOI"].describe()

# Cell 5: CNOI distribution plot
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].hist(cnoi_df["CNOI"], bins=20, edgecolor="black")
ax[0].set_xlabel("CNOI Score")
ax[0].set_ylabel("Frequency")
ax[0].set_title("Distribution of CNOI Scores")

cnoi_df.boxplot(column="CNOI", ax=ax[1])
ax[1].set_ylabel("CNOI Score")
ax[1].set_title("CNOI Boxplot")

plt.tight_layout()
plt.show()

# Cell 6: Top/Bottom banks
top_transparent = cnoi_df.groupby("issuer")["CNOI"].mean().sort_values().head(10)
top_opaque = cnoi_df.groupby("issuer")["CNOI"].mean().sort_values(ascending=False).head(10)

print("=== MOST TRANSPARENT BANKS (Low CNOI) ===")
print(top_transparent)
print("\n=== MOST OPAQUE BANKS (High CNOI) ===")
print(top_opaque)

# Cell 7: Dimension correlation heatmap
dimension_cols = ["D", "G", "R", "J", "T", "S", "X"]
corr_matrix = cnoi_df[dimension_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1)
plt.title("CNOI Dimension Correlations")
plt.show()

# Cell 8: Enrich with tickers
cnoi_df = enrich_cnoi_with_tickers(cnoi_df)
print(f"Ticker coverage: {cnoi_df['ticker'].notna().sum() / len(cnoi_df):.1%}")

# Cell 9: Fetch market data
unique_tickers = cnoi_df["ticker"].dropna().unique().tolist()
print(f"Fetching market data for {len(unique_tickers)} tickers...")

market_df = fetch_bulk_data(
    unique_tickers,
    start_date="2023-01-01",
    end_date="2024-12-31"
)

# Cell 10: Data quality
quality = validate_data_quality(market_df)
pd.DataFrame([quality])

# Cell 11: Save results
cnoi_df.to_csv("../results/cnoi_with_tickers.csv", index=False)
market_df.to_csv("../results/market_returns.csv", index=False)
print("✓ Data saved to results/")
```

**Checkpoint 3.2**: Notebook 1 complete, runs end-to-end

---

### Task 3.3: Notebook 2 - Decile Analysis (6-7 hours)

#### 3.3.1: Create `notebooks/02_decile_analysis.ipynb`

**Structure**:
1. Load pre-processed data (from Notebook 1)
2. Merge CNOI with forward returns
3. Run decile backtest with real data
4. Analyze results:
   - Decile summary table
   - Long-short spread (D1 - D10)
   - Statistical significance (Newey-West t-test)
   - Performance chart (cumulative returns by decile)
5. Sensitivity analysis (different lag days, weighting schemes)
6. Export results

**Key Analysis**:
- Equal-weighted vs value-weighted portfolios
- 2-day vs 5-day lag after filing
- Quarterly rebalancing
- Transaction cost sensitivity

**Checkpoint 3.3**: Notebook 2 complete with real backtest results

---

### Task 3.4: Notebook 3 - Event Study (6-7 hours)

#### 3.4.1: Create `notebooks/03_event_study.ipynb`

**Structure**:
1. Event: SVB collapse (March 9-17, 2023)
2. Load bank data with CNOI scores
3. Fetch market data for event window
4. Compute market model parameters (estimation window: Jan-Feb 2023)
5. Calculate abnormal returns for event window
6. Cumulative abnormal returns (CAR) by CNOI quartile
7. Test hypothesis: High CNOI banks → Worse CAR
8. Visualizations:
   - CAR by CNOI quartile
   - Timeline of abnormal returns
   - Bank-level scatter: CNOI vs CAR

**Real Event Dates**:
- Estimation window: 2023-01-03 to 2023-03-08 (60 days)
- Event window: 2023-03-09 to 2023-03-17 (7 days)
- SVB failure: March 10, 2023

**Checkpoint 3.4**: Notebook 3 complete with SVB event analysis

---

### Task 3.5: Notebook 4 - Panel Regression (6-7 hours)

#### 3.5.1: Create `notebooks/04_panel_regression.ipynb`

**Structure**:
1. Prepare panel dataset (ticker × quarter)
2. Add control variables (log market cap, leverage if available)
3. Run Fixed Effects regression
4. Run Fama-MacBeth regression
5. Run Driscoll-Kraay regression
6. Compare results across methods
7. Dimension-level regressions (which dimensions matter?)
8. Robustness: subsamples, different periods

**Models**:
```
Model 1 (Baseline): ret_t+1 = α + β·CNOI_t + ε
Model 2 (Controls): ret_t+1 = α + β·CNOI_t + γ·log(mcap)_t + ε
Model 3 (Dimensions): ret_t+1 = α + Σ β_i·Dimension_i,t + ε
```

**Checkpoint 3.5**: Notebook 4 complete with panel regression results

---

### Task 3.6: Results Validation & Documentation (2-3 hours)

#### 3.6.1: Validate Results Against Benchmarks

**Checks**:
1. Decile spread magnitude reasonable (<10% annual)
2. Event study CAR reasonable (-5% to -20% for crisis)
3. Panel regression coefficients sensible
4. Results stable across different specifications

#### 3.6.2: Update README with Real Results

Replace hypothetical results with actual findings from notebooks.

#### 3.6.3: Create Results Summary

**File**: `results/README.md`

```markdown
# Analysis Results

## Generated: [Date]

### Decile Backtest (Equal-Weighted, 2-day lag)

| Decile | Mean Return | Std Dev | T-Stat | N Obs |
|--------|-------------|---------|--------|-------|
| D1 (Low CNOI) | X.XX% | X.XX% | X.XX | XXX |
| ...    | ...   | ...   | ...  | ... |
| D10 (High CNOI) | X.XX% | X.XX% | X.XX | XXX |
| **D1-D10** | **X.XX%** | **X.XX%** | **X.XX*** | |

***: p < 0.05

### Event Study (SVB Collapse, March 2023)

| CNOI Quartile | CAR (7-day) | T-Stat |
|---------------|-------------|--------|
| Q1 (Transparent) | -X.X% | -X.XX |
| Q2 | -X.X% | -X.XX |
| Q3 | -X.X% | -X.XX |
| Q4 (Opaque) | -XX.X% | -X.XX** |

### Panel Regression

| Method | CNOI Coefficient | T-Stat | R² |
|--------|------------------|--------|-----|
| Fixed Effects | -0.XXX | -X.XX** | 0.XX |
| Fama-MacBeth | -0.XXX | -X.XX* | N/A |
| Driscoll-Kraay | -0.XXX | -X.XX** | 0.XX |

## Interpretation

[Key findings and implications]

## Data Coverage

- Banks: XX unique tickers
- Filings: XXX CNOI observations
- Time period: 2023-01-01 to 2024-12-31
- Market data completeness: XX%
```

**Checkpoint 3.6**: Results validated and documented

---

## 📊 Definition of Done (Phase 3)

### Notebooks Complete
- [x] `01_data_exploration.ipynb`: Runs end-to-end, visualizations clear
- [x] `02_decile_analysis.ipynb`: Real backtest results generated
- [x] `03_event_study.ipynb`: SVB event analyzed
- [x] `04_panel_regression.ipynb`: All three methods implemented
- [x] All notebooks have markdown explanations

### Real Data Integration
- [x] yfinance fetching working with rate limiting
- [x] Data quality validation passing
- [x] Caching functional (speeds up re-runs)
- [x] Retry logic handles failures gracefully

### Results Generated
- [x] Decile summary table in `results/`
- [x] Event study results in `results/`
- [x] Panel regression output in `results/`
- [x] Results README documenting findings

### Validation
- [x] Results have reasonable magnitudes
- [x] Statistical significance where expected
- [x] Methodology matches academic standards
- [x] Limitations documented

### CI/CD
- [x] All tests still passing
- [x] Notebooks don't break pipeline (use nbconvert for testing)

---

## ✅ Ready for Phase 4 When...

1. ✅ All 4 notebooks run without errors
2. ✅ Real data fetched and cached
3. ✅ Results validated and documented
4. ✅ CI/CD green
5. ✅ README updated with real findings

**Next Phase**: `PHASE4_ADVANCED_FEATURES.md`

---

**Document Control**

**Version**: 1.0
**Status**: 🔴 Blocked (requires Phase 2)
**Estimated Time**: 25-35 hours
**Dependencies**: Phase 2 (all analysis modules)
**Next Phase**: Phase 4 (Advanced Features)
