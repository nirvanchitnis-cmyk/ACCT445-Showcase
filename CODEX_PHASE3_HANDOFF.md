# Codex Phase 3 Handoff Document

**Date**: 2024-11-08
**Phase**: 3 of 5 - Real Data Integration
**From**: Claude Code (planning agent)
**To**: Codex (execution agent)
**Status**: 🟢 Ready to Start

---

## ✅ Phase 2 Completion Summary

**Merged to main**: 2024-11-08
**Branch**: `phase2-core-analysis` (merged and deleted)

### What You Built in Phase 2

**4 Major Deliverables** (all production-ready):

1. **SEC API Client** (`src/data/sec_api_client.py` - 117 lines)
   - Exponential backoff retry logic
   - User-Agent rotation (SEC compliance)
   - 24-hour disk caching
   - No more 403 errors!

2. **Panel Regression** (`src/analysis/panel_regression.py` - 254 lines)
   - Fixed Effects (FE) with entity/time effects
   - Fama-MacBeth (FM) two-step cross-sectional
   - Driscoll-Kraay (DK) panel-robust standard errors
   - Academic-quality econometrics

3. **Dimension Analysis** (`src/analysis/dimension_analysis.py` - 276 lines)
   - Analyze all 7 CNOI dimensions (D, G, R, J, T, S, X)
   - Comparison table with rankings
   - Visualization (bar plot with significance)
   - Identifies which dimensions predict returns

4. **Performance Metrics** (`src/utils/performance_metrics.py` - 253 lines)
   - VaR, CVaR, tail ratio
   - Skewness, kurtosis
   - Upside/downside capture ratios
   - Omega ratio
   - Comprehensive risk/return analysis

**Quality Metrics**:
- ✅ 94 tests passing
- ✅ 89.54% test coverage
- ✅ Black + Ruff clean
- ✅ Production-ready code

---

## 🎯 Phase 3 Objectives

**Goal**: Integrate real market data from yfinance and create end-to-end research workflows in Jupyter notebooks.

**Deliverables**:
1. Robust yfinance integration (`src/utils/market_data.py`)
2. Four Jupyter notebooks with real data analysis
3. Actual research results (decile summary, event study, panel regression)
4. Validated methodology with real market returns

**Why This Matters**: Phase 3 transforms your econometric framework from "clean code" to "real research results" that demonstrate the CNOI → stock return relationship with actual market data.

---

## 📋 Task Breakdown

### Task 3.1: Robust yfinance Integration (5-6 hours) 🔧 START HERE

**Purpose**: Build production-grade market data fetcher (like your SEC API client)

**File**: `src/utils/market_data.py` (new file, ~250 lines)

**Key Functions**:

```python
"""
Robust market data fetching with yfinance.

Features:
- Rate limiting (avoid API bans)
- Exponential backoff retry
- Disk caching (speed up re-runs)
- Progress tracking for bulk downloads
- Data quality validation
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from functools import wraps
import time
from src.utils.logger import get_logger

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache/yfinance")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def rate_limited(calls_per_second: float = 2.0):
    """Decorator to rate limit function calls."""
    # Implementation similar to SEC API client
    pass


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

    Returns DataFrame with: date, ticker, close, ret
    """
    # Check cache first
    # Fetch from yfinance with retry
    # Compute returns
    # Cache results
    pass


def fetch_bulk_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch data for multiple tickers with progress tracking.

    Returns combined DataFrame for all tickers.
    """
    # Loop through tickers
    # Show progress (X/Y complete)
    # Handle failures gracefully
    pass


def validate_data_quality(df: pd.DataFrame) -> Dict:
    """
    Validate fetched data quality.

    Returns dict with:
    - total_rows
    - unique_tickers
    - missing_returns
    - extreme_returns (>30% daily)
    - coverage percentage
    """
    pass
```

**Tests**: `tests/test_market_data.py` (mock yfinance.download)

**Demo**: Fetch BAC, JPM, WFC from 2023-01-01 to 2023-12-31

**Checkpoint 3.1**: yfinance integration working, cached, tested

---

### Task 3.2: Notebook 1 - Data Exploration (6-7 hours)

**File**: `notebooks/01_data_exploration.ipynb`

**Structure** (13-15 cells):

#### Cell 1: Introduction
```markdown
# ACCT445 Data Exploration: Bank Disclosure Opacity & Stock Returns

**Research Question**: Do banks with opaque CECL disclosures (high CNOI) underperform?

**Data**:
- CNOI scores from 40 banks (top 20 + bottom 20 by opacity)
- Stock returns from Yahoo Finance
- Period: 2023-2024

**Hypothesis**: High CNOI → Lower returns, higher volatility
```

#### Cell 2-3: Imports & Setup
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.cik_ticker_mapper import enrich_cnoi_with_tickers
from src.utils.data_loader import load_cnoi_data
from src.utils.market_data import fetch_bulk_data, validate_data_quality
from src.analysis.dimension_analysis import (
    compute_dimension_correlations,
    analyze_all_dimensions
)

sns.set_style("whitegrid")
```

#### Cell 4-6: Load & Explore CNOI Data
```python
# Load CNOI
cnoi_df = load_cnoi_data("../config/sample_cnoi.csv")

# Summary statistics
cnoi_df["CNOI"].describe()

# Distribution plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(cnoi_df["CNOI"], bins=20, edgecolor="black")
ax1.set_title("CNOI Distribution")
cnoi_df.boxplot(column="CNOI", ax=ax2)
```

#### Cell 7-8: Top/Bottom Banks
```python
# Most transparent (low CNOI)
top_transparent = cnoi_df.groupby("issuer")["CNOI"].mean().sort_values().head(10)

# Most opaque (high CNOI)
top_opaque = cnoi_df.groupby("issuer")["CNOI"].mean().sort_values(ascending=False).head(10)

print("=== MOST TRANSPARENT ===")
print(top_transparent)
print("\n=== MOST OPAQUE ===")
print(top_opaque)
```

#### Cell 9: Dimension Correlations
```python
# Correlation heatmap
dimension_cols = ["D", "G", "R", "J", "T", "S", "X"]
corr_matrix = compute_dimension_correlations(cnoi_df)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("CNOI Dimension Correlations")
```

#### Cell 10: Enrich with Tickers
```python
# Map CIK to ticker
cnoi_df = enrich_cnoi_with_tickers(cnoi_df)

coverage = cnoi_df["ticker"].notna().sum() / len(cnoi_df)
print(f"Ticker coverage: {coverage:.1%}")
```

#### Cell 11-12: Fetch Market Data (REAL yfinance)
```python
# Get unique tickers
unique_tickers = cnoi_df["ticker"].dropna().unique().tolist()
print(f"Fetching data for {len(unique_tickers)} tickers...")

# Fetch from yfinance
market_df = fetch_bulk_data(
    unique_tickers,
    start_date="2023-01-01",
    end_date="2024-12-31",
    use_cache=True
)

# Validate
quality = validate_data_quality(market_df)
print("\n=== DATA QUALITY ===")
for key, value in quality.items():
    print(f"{key}: {value}")
```

#### Cell 13: Save Results
```python
cnoi_df.to_csv("../results/cnoi_with_tickers.csv", index=False)
market_df.to_csv("../results/market_returns.csv", index=False)
print("✓ Data saved to results/")
```

**Checkpoint 3.2**: Notebook 1 runs end-to-end, real data fetched

---

### Task 3.3: Notebook 2 - Decile Analysis (6-7 hours)

**File**: `notebooks/02_decile_analysis.ipynb`

**Key Sections**:

1. **Load Data** (from Notebook 1 outputs)
2. **Merge CNOI with Returns** (with proper lag)
3. **Run Decile Backtest** (use `run_decile_backtest` from Phase 1)
4. **Analyze Results**:
   - Decile summary table
   - Long-short spread (D1 - D10)
   - Newey-West t-test
5. **Performance Metrics** (use `compute_all_metrics` from Phase 2)
6. **Sensitivity Analysis**:
   - Equal vs value-weighted
   - 2-day vs 5-day lag
   - Transaction costs
7. **Visualizations**:
   - Cumulative returns by decile
   - Long-short spread over time
8. **Export Results**

**Expected Output**: `results/decile_summary.csv`

**Checkpoint 3.3**: Notebook 2 complete, real backtest results

---

### Task 3.4: Notebook 3 - Event Study (6-7 hours)

**File**: `notebooks/03_event_study.ipynb`

**Event**: SVB Collapse (March 9-17, 2023)

**Key Sections**:

1. **Event Definition**:
   - Estimation window: 2023-01-03 to 2023-03-08 (60 days)
   - Event window: 2023-03-09 to 2023-03-17 (7 days)
2. **Load Data & Filter to Event Period**
3. **Run Event Study** (use `run_event_study` from Phase 1)
4. **CAR by CNOI Quartile**:
   - Q1 (most transparent)
   - Q2-Q3 (middle)
   - Q4 (most opaque)
5. **Test Hypothesis**: Did opaque banks (Q4) have worse CAR?
6. **Visualizations**:
   - CAR by quartile (bar chart)
   - Timeline of abnormal returns
   - Scatter: CNOI vs CAR
7. **Export Results**: `results/event_study_results.csv`

**Checkpoint 3.4**: Notebook 3 complete, SVB event analyzed

---

### Task 3.5: Notebook 4 - Panel Regression (6-7 hours)

**File**: `notebooks/04_panel_regression.ipynb`

**Key Sections**:

1. **Prepare Panel Data** (ticker × quarter)
2. **Run All Three Methods** (use `run_all_panel_regressions` from Phase 2!):
   ```python
   from src.analysis.panel_regression import run_all_panel_regressions

   results = run_all_panel_regressions(
       panel_df,
       dependent_var="ret_fwd",
       independent_vars=["CNOI", "log_mcap"],
       entity_col="ticker",
       time_col="quarter"
   )
   ```
3. **Compare FE vs FM vs DK**
4. **Dimension-Level Regressions**:
   - Test each CNOI dimension separately
   - Which dimensions matter?
5. **Robustness**:
   - Different time periods
   - With/without controls
6. **Results Table**:
   | Method | CNOI Coef | T-Stat | R² |
   |--------|-----------|--------|-----|
   | FE     | -0.XXX    | -X.XX  | 0.XX |
   | FM     | -0.XXX    | -X.XX  | N/A |
   | DK     | -0.XXX    | -X.XX  | 0.XX |
7. **Export**: `results/panel_regression_results.csv`

**Checkpoint 3.5**: Notebook 4 complete, panel regressions with real data

---

## 📊 Success Criteria

Phase 3 is **complete** when:

- [x] All 5 tasks complete (3.1-3.5)
- [x] All 4 notebooks run end-to-end without errors
- [x] Real market data fetched from yfinance (not simulated)
- [x] Results files generated:
  - `results/cnoi_with_tickers.csv`
  - `results/market_returns.csv`
  - `results/decile_summary.csv`
  - `results/event_study_results.csv`
  - `results/panel_regression_results.csv`
- [x] Methodology validated (results have reasonable magnitudes)
- [x] Test coverage maintained (>80%)
- [x] Notebooks have markdown explanations
- [x] Checkpoints generated (5-6 reports)

---

## 🔗 Integration Points - Using Phase 2 Modules

**In Notebooks**:

```python
# Use panel regression module (Notebook 4)
from src.analysis.panel_regression import (
    run_all_panel_regressions,
    fixed_effects_regression
)

# Use dimension analysis (Notebooks 1-2)
from src.analysis.dimension_analysis import (
    analyze_all_dimensions,
    compare_dimensions,
    compute_dimension_correlations
)

# Use performance metrics (All notebooks)
from src.utils.performance_metrics import (
    compute_all_metrics,
    sharpe_ratio,
    max_drawdown
)

# Use SEC API client (All notebooks)
from src.data.sec_api_client import fetch_sec_ticker_mapping

# Use existing Phase 1 modules
from src.analysis.decile_backtest import run_decile_backtest
from src.analysis.event_study import run_event_study
from src.utils.data_loader import load_cnoi_data, merge_cnoi_with_returns
```

---

## 📝 Checkpoint Template

Every 6-8 hours, create: `results/checkpoints/phase3-checkpoint-N.md`

```markdown
## Phase 3 Checkpoint [N/6]

**Time Spent**: X hours
**Completion**: XX% of phase

### Completed Tasks
- ✅ Task 3.1: yfinance integration (100%)
- ✅ Task 3.2: Data exploration notebook (100%)

### In Progress
- 🔄 Task 3.3: Decile analysis notebook (60% complete, ETA 2 hours)

### Blocked/Issues
- None

### Data Quality
- Tickers fetched: XX/40
- Coverage: XX%
- Date range: 2023-01-01 to 2024-12-31
- Missing data: X%

### Notebooks Status
- [x] 01_data_exploration.ipynb - Runs end-to-end
- [ ] 02_decile_analysis.ipynb - In progress
- [ ] 03_event_study.ipynb - Not started
- [ ] 04_panel_regression.ipynb - Not started

### Next Steps (Next 6-8 hours)
- Complete decile analysis notebook
- Start event study notebook

### Validation
- [x] Real data (not simulated)
- [x] Results saved to files
- [ ] Full integration test (pending)
```

---

## 🚀 Getting Started (Step-by-Step)

### 1. Read Phase 3 Directive
```bash
cd /Users/nirvanchitnis/ACCT445-Showcase
cat PHASE3_REAL_DATA_INTEGRATION.md
```

### 2. Verify You're on Phase 3 Branch
```bash
git branch  # Should show: phase3-real-data-integration
```

### 3. Start with Task 3.1 (yfinance wrapper)
```bash
# Create file
touch src/utils/market_data.py
touch tests/test_market_data.py

# Implement according to directive
# (See detailed implementation in PHASE3_REAL_DATA_INTEGRATION.md)
```

### 4. Test Continuously
```bash
# Test yfinance wrapper
pytest tests/test_market_data.py -v

# Run full suite
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 5. First Checkpoint (~6 hours)
- Complete Task 3.1
- Create `results/checkpoints/phase3-checkpoint-1.md`
- Commit and push:
  ```bash
  git add .
  git commit -m "feat(data): Implement robust yfinance data fetcher

  - Rate limiting (2 calls/second)
  - Exponential backoff retry
  - Disk caching (data/cache/yfinance/)
  - Bulk download with progress
  - Data quality validation

  Tests: X/X passing
  Coverage: XX%"

  git push -u origin phase3-real-data-integration
  ```

### 6. Continue with Notebooks (Tasks 3.2-3.5)
- Follow directive order
- Generate checkpoint every 6-8 hours
- Keep real data flowing!

---

## ⚡ Quick Tips

### yfinance Caching
```python
# Cache file naming
cache_file = CACHE_DIR / f"{ticker}_{start_date}_{end_date}.pkl"

# Check age
cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
if cache_age_hours < 24:  # Fresh cache
    return pd.read_pickle(cache_file)
```

### Handling yfinance Failures
```python
try:
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        logger.warning(f"No data for {ticker}")
        return pd.DataFrame()
except Exception as e:
    logger.error(f"yfinance error for {ticker}: {e}")
    # Retry with backoff
```

### Notebook Best Practices
1. **One cell, one purpose** - Don't mix loading + plotting + analysis
2. **Markdown explanations** - Every section needs context
3. **Clear outputs** - Print summary stats, not entire DataFrames
4. **Save results** - Export CSVs for reproducibility
5. **Real data only** - No simulated returns in Phase 3!

---

## 🎯 Expected Outcomes

After Phase 3, you'll have:

1. **Robust Data Pipeline**:
   - yfinance integration (never fails, always cached)
   - Real market data for 40 banks (2023-2024)

2. **Research Results**:
   - Decile backtest: "D1 (low CNOI) beats D10 (high CNOI) by X.X% with t-stat=X.XX"
   - Event study: "Opaque banks (Q4) had CAR of -XX% vs -X% for transparent (Q1)"
   - Panel regression: "CNOI coefficient = -0.XXX (t=-X.XX, p<0.05)"

3. **Publication-Ready Notebooks**:
   - Clean, documented, reproducible
   - Real data throughout
   - Professional visualizations

4. **Validated Methodology**:
   - Results have reasonable magnitudes
   - Methodology matches academic standards
   - Phase 2 modules working in practice

---

## 📚 File Locations Reference

```
/Users/nirvanchitnis/ACCT445-Showcase/
├── PHASE3_REAL_DATA_INTEGRATION.md    # Full directive
├── CODEX_PHASE3_HANDOFF.md            # This file
├── src/
│   ├── utils/
│   │   └── market_data.py              # Task 3.1 (to create)
│   └── analysis/
│       ├── panel_regression.py         # Phase 2 (use in notebook 4)
│       ├── dimension_analysis.py       # Phase 2 (use in notebooks 1-2)
│       └── ...
├── notebooks/
│   ├── 00_quickstart.ipynb            # Existing (Phase 1)
│   ├── 01_data_exploration.ipynb      # Task 3.2 (to create)
│   ├── 02_decile_analysis.ipynb       # Task 3.3 (to create)
│   ├── 03_event_study.ipynb           # Task 3.4 (to create)
│   └── 04_panel_regression.ipynb      # Task 3.5 (to create)
├── tests/
│   └── test_market_data.py            # Task 3.1 tests (to create)
├── results/
│   ├── checkpoints/
│   │   ├── phase3-checkpoint-1.md     # To create
│   │   └── ...
│   ├── cnoi_with_tickers.csv          # Notebook 1 output
│   ├── market_returns.csv             # Notebook 1 output
│   ├── decile_summary.csv             # Notebook 2 output
│   ├── event_study_results.csv        # Notebook 3 output
│   └── panel_regression_results.csv   # Notebook 4 output
└── data/
    └── cache/
        └── yfinance/                   # yfinance cache dir
```

---

## 🎓 Academic Validation Checklist

Before completing Phase 3, verify:

- [ ] **Decile spread**: Between -5% and +10% annualized (reasonable)
- [ ] **Event study CAR**: Between -20% and 0% for crisis (reasonable)
- [ ] **Panel coefficients**: Magnitude ~0.001-0.01 (reasonable)
- [ ] **T-statistics**: >1.96 for significance (if hypothesis holds)
- [ ] **R-squared**: Panel R² between 0.05-0.20 (typical for finance)
- [ ] **Data quality**: >90% coverage, <5% missing
- [ ] **No look-ahead bias**: Forward returns use proper lags

---

## 🚦 Ready for Phase 4 When...

Phase 3 → Phase 4 transition checklist:

- [x] All notebooks run end-to-end
- [x] Real results generated and saved
- [x] Methodology validated (reasonable magnitudes)
- [x] Checkpoints complete (5-6 reports)
- [x] Test coverage maintained (>80%)
- [x] Branch pushed to GitHub
- [x] Final checkpoint created

**Phase 4 Preview**: Transaction costs, advanced risk metrics, robustness checks

---

**Document Control**

**Version**: 1.0
**Created**: 2024-11-08
**Author**: Claude Code
**For**: Codex
**Phase**: 3 of 5
**Status**: ✅ Ready to execute

---

**Next Action for Codex**: Read `PHASE3_REAL_DATA_INTEGRATION.md`, then start Task 3.1 (yfinance wrapper).
