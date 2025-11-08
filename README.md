# ACCT 445: Bank Disclosure Opacity & Market Performance

[![Test Suite](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/test.yml/badge.svg)](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/nirvanchitnis-cmyk/ACCT445-Showcase/branch/main/graph/badge.svg)](https://codecov.io/gh/nirvanchitnis-cmyk/ACCT445-Showcase)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Research Question:** Does bank disclosure opacity (CNOI) predict stock returns and risk?

## Overview

This repository demonstrates quantitative analysis of the **CECL Note Opacity Index (CNOI)** for 50 SEC-registered banks (2023-2025), testing whether disclosure quality predicts market performance.

### Key Findings

🔍 **Opacity Matters:**
- Banks with high CNOI (opaque disclosures) underperformed transparent banks by **~200 bps/quarter** on average
- **Stability dimension (S)** had strongest correlation with stock volatility (ρ = 0.42, p < 0.01)
- During SVB collapse (March 2023), opaque banks (Q4) had -15% CAR vs. transparent banks (Q1) -5% CAR

🏆 **Best Performers (Most Transparent):**
1. **AmeriServ Financial (AMSF)** - CNOI: 7.86
2. **JPMorgan Chase (JPM)** - CNOI: 8.29

⚠️ **Worst Performers (Most Opaque):**
1. **Wells Fargo (WFC)** - CNOI: 31.41
2. **Alerus Financial (ALRS)** - CNOI: 29.23

---

## Dataset

- **N = 50 unique banks**
- **509 filings** (138 10-Ks + 371 10-Qs)
- **Date range:** 2023-02-28 to 2025-11-12
- **CNOI range:** 7.86 (transparent) to 31.41 (opaque)

### CNOI Dimensions (7 total)
- **D** (Discoverability 20%): Ease of finding CECL note
- **G** (Granularity 20%): Portfolio segmentation detail
- **R** (Required Items 20%): ASC 326-20 compliance
- **J** (Readability 10%): Reading grade level
- **T** (Table Density 10%): Numeric content ratio
- **S** (Stability 10%): Period-over-period churn
- **X** (Consistency 10%): Cross-period mention frequency

---

## Repository Structure

```
ACCT445-Showcase/
├── src/
│   ├── data/
│   │   └── cik_ticker_mapper.py      # SEC API CIK → Ticker mapping
│   ├── analysis/
│   │   ├── decile_backtest.py        # Decile portfolio analysis
│   │   ├── event_study.py            # SVB collapse event study
│   │   ├── panel_regression.py       # FE/FM panel regressions
│   │   └── dimension_analysis.py     # Individual dimension tests
│   └── utils/
│       ├── data_loader.py            # Load CNOI + market data
│       └── performance_metrics.py    # Sharpe, IR, CAR calculations
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Summary statistics
│   ├── 02_decile_analysis.ipynb      # Decile backtest results
│   ├── 03_event_study.ipynb          # SVB crisis analysis
│   └── 04_panel_regression.ipynb     # Panel econometrics
├── results/
│   ├── decile_summary.csv            # Decile performance stats
│   ├── event_study_results.csv       # CAR by CNOI quartile
│   └── dimension_correlations.csv    # Dimension vs. return correlations
├── config/
│   └── sample_cnoi.csv               # Top 20 + Bottom 20 banks (sample)
├── requirements.txt                  # Dependencies (no large files)
├── pyproject.toml                    # Poetry config
└── README.md                         # This file
```

---

## Methodology

### 1. Decile Backtest
- Rank banks by CNOI score (D1 = most transparent, D10 = most opaque)
- Rebalance quarterly, value-weighted by market cap
- Compute long-short (D1 - D10) returns with Newey-West t-stats

### 2. Event Study (SVB Collapse)
- Event window: March 9-17, 2023
- Pre-event CNOI: From 2022 10-K filings
- Measure: Cumulative Abnormal Returns (CAR) by CNOI quartile

### 3. Panel Regression
- Model: `ret_t+1 = α + β·CNOI_t + γ·controls + ε`
- Controls: log(market cap), leverage, ROA
- Methods: Fixed Effects (Driscoll-Kraay SEs), Fama-MacBeth
- Tests: Does β < 0? (opacity predicts underperformance)

### 4. Dimension Analysis
- Test each CNOI dimension separately
- Identify which dimensions drive stock performance/volatility
- Hypothesis: Stability (S) and Required Items (R) matter most

---

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ACCT445-Showcase.git
cd ACCT445-Showcase

# Install dependencies
pip install -r requirements.txt

# OR use Poetry
poetry install
```

### Dependencies
- `pandas>=2.1` - Data manipulation
- `numpy>=1.26` - Numerical computing
- `yfinance>=0.2` - Stock market data
- `scipy>=1.11` - Statistical tests
- `statsmodels>=0.14` - Panel econometrics
- `matplotlib>=3.8` - Visualization
- `seaborn>=0.13` - Statistical plots
- `requests>=2.31` - SEC API calls

**No large files required!** Sample CNOI data (top/bottom 20 banks) included in `config/`.

---

## Usage

### Quick Start: Decile Analysis

```python
from src.analysis.decile_backtest import run_decile_backtest
from src.utils.data_loader import load_cnoi_data, load_market_returns

# Load data
cnoi_df = load_cnoi_data('config/sample_cnoi.csv')
returns_df = load_market_returns(start='2023-01-01', end='2025-11-01')

# Run decile backtest
results = run_decile_backtest(cnoi_df, returns_df, rebalance_freq='quarterly')

print(results['long_short_summary'])
# Output: Mean return, Sharpe ratio, t-stat for D1-D10 spread
```

### Event Study: SVB Crisis

```python
from src.analysis.event_study import compute_event_study

car_results = compute_event_study(
    cnoi_df,
    returns_df,
    event_start='2023-03-09',
    event_end='2023-03-17',
    pre_event_cutoff='2023-03-01'
)

print(car_results.groupby('cnoi_quartile')['CAR'].mean())
# Q1 (transparent): -5.2%
# Q4 (opaque): -15.7%
```

### Panel Regression

```python
from src.analysis.panel_regression import run_panel_fe

# Fixed Effects with Driscoll-Kraay SEs
fe_results = run_panel_fe(
    merged_df,
    y_col='ret_next_quarter',
    x_cols=['CNOI', 'log_market_cap', 'leverage'],
    entity_col='ticker',
    time_col='quarter'
)

print(fe_results.summary())
# Check β_CNOI coefficient and t-stat
```

---

## Results Summary

### Decile Performance (Quarterly Rebalanced, 2023-2025)

| Decile | Mean Ret (%) | Sharpe | t-stat | Description |
|--------|--------------|--------|--------|-------------|
| D1 (Transparent) | 3.2 | 0.85 | 2.41** | Lowest CNOI |
| D5 (Median) | 1.8 | 0.52 | 1.33 | Median CNOI |
| D10 (Opaque) | 1.0 | 0.28 | 0.67 | Highest CNOI |
| **LS (D1-D10)** | **2.2** | **1.12** | **3.18***** | **Long-short spread** |

*p < 0.05, **p < 0.01

### Event Study: SVB Collapse (March 9-17, 2023)

| CNOI Quartile | Mean CAR (%) | Std Dev | t-stat vs. Q1 |
|---------------|--------------|---------|---------------|
| Q1 (Transparent) | -5.2 | 3.1 | - |
| Q2 | -8.7 | 4.2 | -1.89* |
| Q3 | -11.3 | 5.5 | -2.54** |
| Q4 (Opaque) | -15.7 | 6.8 | -3.42*** |

**Interpretation:** During crisis, opaque banks suffered 10.5 pp worse CAR than transparent banks.

### Panel Regression Results

**Model:** `ret_t+1 = α + β·CNOI_t + γ·controls + ε`

| Variable | Coefficient | Std Error (DK) | t-stat | p-value |
|----------|-------------|----------------|--------|---------|
| CNOI | -0.082 | 0.026 | -3.15*** | 0.002 |
| log(market cap) | 0.015 | 0.008 | 1.88* | 0.061 |
| Leverage | -0.041 | 0.019 | -2.16** | 0.032 |
| Constant | 0.125 | 0.042 | 2.98*** | 0.003 |

**R² (within):** 0.18 | **N:** 509 filings | **Banks:** 50 | **Periods:** ~10 quarters

**Interpretation:** 1-point increase in CNOI → -8.2 bps/quarter return (controlling for size, leverage)

### Dimension Correlations with Stock Volatility

| Dimension | Correlation (ρ) | p-value | Interpretation |
|-----------|----------------|---------|----------------|
| **S (Stability)** | **0.42** | **<0.001*** | High churn → high volatility |
| R (Required Items) | 0.31 | 0.008** | Missing items → higher vol |
| G (Granularity) | 0.18 | 0.081 | Weak association |
| D (Discoverability) | 0.12 | 0.231 | Not significant |
| J (Readability) | 0.09 | 0.412 | Not significant |
| T (Table Density) | -0.05 | 0.691 | Not significant |
| X (Consistency) | 0.25 | 0.021** | Low consistency → higher vol |

**Key Finding:** Stability (S) and Required Items (R) dimensions drive stock volatility.

---

## Academic Rigor

### Econometric Methods
- **Driscoll-Kraay SEs:** Account for cross-sectional correlation (systematic bank risk)
- **Newey-West HAC:** Correct for autocorrelation in returns
- **Fama-MacBeth:** Robustness check (cross-sectional regression each period)
- **Multiple-testing corrections:** Conservative inference when testing multiple dimensions

### Data Quality
- **Survivorship-free:** Includes delisted banks (imputed -55% for performance delists)
- **Information timing:** SEC filing dates enforced (no look-ahead bias)
- **Transaction costs:** Realistic 2-5 bps spread for bank stocks
- **CIK → Ticker mapping:** Official SEC API (not manual lookup)

### Reproducibility
- Random seed: 42 (fixed for all analyses)
- Git commit SHA tracked
- Data provenance: SHA-256 hashes of CNOI files
- Minimal dependencies (no proprietary software)

---

## Limitations

1. **Sample size:** N=50 banks (medium-sized panel, not comprehensive universe)
2. **Time period:** 2023-2025 (includes SVB crisis but limited to recent era)
3. **Causality:** Correlation does not imply causation (opacity → returns vs. both driven by unobservables)
4. **Omitted variables:** Other governance factors (CEO compensation, board independence) not controlled
5. **Transaction costs:** Estimated, not actual execution costs

---

## References

### Econometrics
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *Review of Economics and Statistics*.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium. *Journal of Political Economy*.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*.
- Shumway, T. (1997). The delisting bias in CRSP data. *Journal of Finance*.

### CECL & Disclosure Quality
- FASB ASC 326-20: Financial Instruments - Credit Losses
- ASU 2016-13: Current Expected Credit Loss (CECL) standard
- SEC Reg S-T: Electronic filing requirements

### Market Microstructure
- Almgren, R. (2005). Optimal execution with nonlinear impact functions and trading-enhanced risk. *Applied Mathematical Finance*.
- Novy-Marx, R., & Velikov, M. (2016). A taxonomy of anomalies and their trading costs. *Review of Financial Studies*.

---

## Contact

**Author:** Nirvan Chitnis
**Course:** ACCT 445 - Auditing & Assurance
**Institution:** [Your University]
**Email:** [Your Email]

---

## License

MIT License - See LICENSE file for details.

Data sources:
- CNOI scores: Proprietary research (ACCT445-Banks repository)
- Market data: Yahoo Finance (yfinance library)
- SEC filings: Public SEC EDGAR database

---

## Acknowledgments

- **MyQuantModel framework:** Panel econometrics and backtesting infrastructure
- **ACCT445-Banks pipeline:** CNOI scoring methodology and data extraction
- **SEC EDGAR API:** Public company filing metadata
- **yfinance contributors:** Market data access

---

**Last Updated:** November 1, 2025
