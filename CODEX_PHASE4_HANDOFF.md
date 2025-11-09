# Codex Phase 4 Handoff Document

**Date**: 2025-11-08
**Phase**: 4 of 5 - Advanced Features & Production Readiness
**From**: Claude Code (planning agent)
**To**: Codex (execution agent)
**Status**: 🟢 Ready to Start

---

## ✅ Phase 3 Completion Summary

**Merged to main**: 2025-11-08
**Achievements**:
- ✅ 103 tests passing (up from 94 in Phase 2)
- ✅ 88.78% test coverage (maintained >80% threshold)
- ✅ Robust yfinance integration (`src/utils/market_data.py`)
- ✅ 100% ticker coverage (40/40 banks mapped with overrides)
- ✅ 4 Jupyter notebooks executed end-to-end:
  - `notebooks/01_data_exploration.ipynb` - Data quality & EDA
  - `notebooks/02_decile_analysis.ipynb` - Decile backtests + dimension analysis
  - `notebooks/03_event_study.ipynb` - SVB collapse event study
  - `notebooks/04_panel_regression.ipynb` - FE/FM/DK panel regressions
- ✅ Real market data integration complete
- ✅ Academic-quality findings documented

**Branch**: `phase3-real-data-integration` (merged and available for deletion)
**Checkpoint Report**: `results/checkpoints/phase3-final.md`

**Key Findings from Phase 3**:
- **Decile Analysis**: Required Items (R) and Consistency (X) dimensions strongest predictors
- **Event Study**: Non-monotonic crisis impact (Q3 suffered most during SVB collapse)
- **Panel Regression**: Discoverability (D) highly significant (t=-4.77), CNOI baseline -17 bps

**Validation**:
```bash
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80
# Result: 103 tests passing, 88.78% coverage ✅

black --check src/ tests/
# Result: All formatted ✅

ruff check src/ tests/
# Result: No errors ✅
```

---

## 🎯 Phase 4 Objectives

**Directive File**: `PHASE4_ADVANCED_FEATURES.md`
**Estimated Time**: 35-45 hours
**Branch**: `phase4-advanced-features` (create new)

Transform from research tool to **production trading system**:

### Deliverables

1. **Transaction Cost Modeling** (8-10 hours)
   - Bid-ask spread estimation (function of volatility, liquidity)
   - Market impact (Almgren-Chriss model)
   - Fixed trading fees
   - Apply costs to backtest returns (realistic 2-5 bps per trade)
   - File: `src/utils/transaction_costs.py`
   - Tests: `tests/test_transaction_costs.py`

2. **Advanced Risk Metrics** (8-10 hours)
   - Already started in Phase 2, extend further
   - Add: Rolling volatility, factor exposures, tail metrics
   - Validate against industry standards (QuantLib)
   - File: Extend `src/utils/performance_metrics.py`
   - Tests: Extend `tests/test_performance_metrics.py`

3. **Robustness Framework** (8-10 hours)
   - Bootstrap resampling (1000 samples, confidence intervals)
   - Permutation tests (significance testing)
   - Subsample analysis (time periods, market regimes)
   - Monte Carlo simulations
   - File: `src/analysis/robustness.py`
   - Tests: `tests/test_robustness.py`

4. **Configuration Management** (4-5 hours)
   - Centralized TOML config file
   - All parameters configurable (no hardcoded values)
   - File: `config/config.toml`
   - File: `src/utils/config.py`
   - Tests: `tests/test_config.py`

5. **Data Versioning (DVC)** (4-5 hours)
   - Install and configure DVC
   - Track data files (CNOI, cache, results)
   - Document DVC workflow in README
   - Files: `.dvc/`, `*.dvc` files

6. **Performance Optimization** (6-8 hours)
   - Disk caching decorator
   - Vectorization (replace loops with pandas ops)
   - Parallelization (joblib for ticker fetching)
   - Target: >50% speedup in backtests
   - File: `src/utils/caching.py`
   - Update: `src/utils/market_data.py` for parallelization

7. **Pre-commit Hooks** (2-3 hours)
   - Automated quality enforcement
   - Black, ruff, trailing whitespace, YAML check
   - Pytest with coverage requirement
   - File: `.pre-commit-config.yaml`
   - Install: `pre-commit install`

### Success Criteria

- ✅ Transaction costs reduce backtest returns by realistic 2-5 bps per trade
- ✅ Advanced risk metrics validated against industry standards
- ✅ Robustness checks confirm main results hold (bootstrap CIs, permutation p-values)
- ✅ All parameters configurable via TOML (no hardcoded values)
- ✅ Data versioned and tracked with DVC
- ✅ Backtest speed improved >50%
- ✅ Pre-commit hooks enforcing quality (black, ruff, pytest)
- ✅ >80% test coverage maintained
- ✅ All tests passing
- ✅ CI/CD pipeline green

---

## 📋 Task Execution Order

### Task 4.1: Transaction Cost Modeling (8-10 hours) 🔧 START HERE

**Why First**: Essential for realistic backtest results. Must be in place before optimizing other components.

**Implementation**: See `PHASE4_ADVANCED_FEATURES.md` lines 34-183

**Key Functions**:
- `estimate_bid_ask_spread(volatility, market_cap, base_spread_bps=5.0)`: Volatility + size-dependent spread
- `compute_market_impact(trade_value, avg_daily_volume, impact_coefficient=0.1)`: Almgren-Chriss market impact
- `apply_transaction_costs(backtest_returns, turnover, avg_spread_bps, commission_bps, avg_impact_bps)`: Apply to backtest

**Formula**:
```
Total cost per trade (one-way) = (spread/2) + commission + impact
Cost per period = turnover × total_cost_bps / 10000
Net return = Gross return - Cost per period
```

**Validation**:
- Typical parameters: 5 bps spread, 1 bps commission, 2 bps impact
- 50% quarterly turnover → ~2 bps cost per quarter
- Annualized cost drag: ~8 bps (0.08%)

**Demo Section**: Simulate backtest returns, apply costs, show gross vs net

**Tests**:
- Test spread estimation with different volatilities
- Test market impact with different trade sizes
- Test cost application to DataFrame
- Edge cases: zero turnover, negative returns

**Checkpoint**: Create `results/checkpoints/phase4-checkpoint-1.md` after completion

---

### Task 4.2: Advanced Risk Metrics (8-10 hours)

**Implementation**: Extend existing `src/utils/performance_metrics.py` (253 lines from Phase 2)

**New Functions to Add**:
- `rolling_volatility(returns, window=21)`: 21-day rolling volatility for vol targeting
- **Already implemented in Phase 2** (verify and document):
  - `value_at_risk(returns, confidence=0.95)`
  - `conditional_var(returns, confidence=0.95)`
  - `tail_ratio(returns)`
  - `skewness(returns)`
  - `kurtosis(returns)`
  - `downside_capture(returns, benchmark)`
  - `upside_capture(returns, benchmark)`
  - `omega_ratio(returns, threshold=0)`

**Validation Strategy**:
- Compare VaR/CVaR against scipy.stats percentiles
- Compare skewness/kurtosis against scipy.stats functions
- Compare Sharpe against manual calculation
- Document any differences and explain

**Checkpoint**: Create `results/checkpoints/phase4-checkpoint-2.md`

---

### Task 4.3: Robustness Framework (8-10 hours)

**Implementation**: See `PHASE4_ADVANCED_FEATURES.md` lines 254-477

**Key Functions**:
- `bootstrap_backtest(df, score_col, n_bootstrap=1000, confidence=0.95)`: Resample with replacement
  - Returns: `{mean, ci_lower, ci_upper, distribution}`
  - Use tqdm for progress bar
- `permutation_test(df, score_col, n_permutations=1000)`: Shuffle labels to test significance
  - Returns: `{observed, p_value, null_distribution}`
- `subsample_analysis(df, score_col, split_col, split_values)`: Test across subsamples
  - Returns: DataFrame with results for each split
  - Example: Split by year, market regime, sector

**Integration with Phase 1-3**:
- Use `run_decile_backtest()` from `src.analysis.decile_backtest`
- Apply to CNOI + all 7 dimensions
- Test findings from Phase 3 (are R, X, D robustly significant?)

**Computational Note**: 1000 bootstrap + 1000 permutation = ~2000 backtest runs
- Optimize by caching intermediate results
- Use `numpy.random.seed()` for reproducibility

**Demo Section**:
- Bootstrap confidence intervals for CNOI long-short spread
- Permutation test for significance
- Subsample by year (2023 vs 2024)

**Checkpoint**: Create `results/checkpoints/phase4-checkpoint-3.md`

---

### Task 4.4: Configuration Management (4-5 hours)

**Implementation**: See `PHASE4_ADVANCED_FEATURES.md` lines 483-570

**File 1: `config/config.toml`**
```toml
[general]
random_seed = 42
log_level = "INFO"

[data]
cnoi_file = "config/sample_cnoi.csv"
cache_dir = "data/cache"
results_dir = "results"

[market_data]
start_date = "2023-01-01"
end_date = "2024-12-31"
rate_limit_calls_per_second = 2.0
max_retries = 3
use_cache = true

[backtest]
n_deciles = 10
weighting = "equal"  # or "value"
lag_days = 2
rebalance_frequency = "Q"

[transaction_costs]
avg_spread_bps = 5.0
commission_bps = 1.0
avg_impact_bps = 2.0
assumed_turnover = 0.5

[risk_metrics]
var_confidence = 0.95
periods_per_year = 252
risk_free_rate = 0.03

[robustness]
n_bootstrap = 1000
n_permutations = 1000
bootstrap_confidence = 0.95

[panel_regression]
entity_effects = true
time_effects = true
cluster_by_entity = true
max_lags = 4
```

**File 2: `src/utils/config.py`**
```python
import toml
from pathlib import Path
from typing import Dict, Any

CONFIG_FILE = Path(__file__).parent.parent.parent / "config" / "config.toml"

def load_config(config_path: Path = CONFIG_FILE) -> Dict[str, Any]:
    """Load configuration from TOML file."""
    return toml.load(config_path)

def get_config_value(key_path: str, default=None) -> Any:
    """
    Get config value using dot notation.

    Example:
        >>> n_deciles = get_config_value("backtest.n_deciles")
        >>> lag_days = get_config_value("backtest.lag_days", default=2)
    """
    config = load_config()
    keys = key_path.split(".")
    value = config
    for key in keys:
        value = value.get(key, default)
        if value is default:
            break
    return value
```

**Migration Task**: Update existing code to use config
- Replace hardcoded `n_deciles=10` with `get_config_value("backtest.n_deciles")`
- Replace hardcoded dates with config values
- Update notebooks to load config at top

**Dependencies**: Add `toml` to requirements.txt

**Tests**:
- Test config loading
- Test get_config_value with nested keys
- Test defaults
- Test missing config file handling

**Checkpoint**: Create `results/checkpoints/phase4-checkpoint-4.md`

---

### Task 4.5: Data Versioning with DVC (4-5 hours)

**Implementation**: See `PHASE4_ADVANCED_FEATURES.md` lines 575-593

**Steps**:
1. Install DVC: `pip install dvc`
2. Initialize: `dvc init`
3. Track data files:
   ```bash
   dvc add config/sample_cnoi.csv
   dvc add data/cache/
   dvc add results/
   ```
4. Commit .dvc files to git:
   ```bash
   git add config/.gitignore config/sample_cnoi.csv.dvc
   git add data/.gitignore data/cache.dvc
   git add results/.gitignore results.dvc
   git commit -m "Add DVC tracking for data files"
   ```

**Update README.md**:
```markdown
## Data Versioning

This project uses DVC (Data Version Control) to track large data files.

### Setup
```bash
pip install dvc
dvc pull  # Download data from remote (if configured)
```

### Updating Data
```bash
# After modifying data files:
dvc add config/sample_cnoi.csv
git add config/sample_cnoi.csv.dvc
git commit -m "Update CNOI data"
```

**Dependencies**: Add `dvc>=3.0.0` to requirements.txt

**Checkpoint**: Create `results/checkpoints/phase4-checkpoint-5.md`

---

### Task 4.6: Performance Optimization (6-8 hours)

**Implementation**: See `PHASE4_ADVANCED_FEATURES.md` lines 595-646

**Three-Pronged Approach**:

1. **Disk Caching** (`src/utils/caching.py`):
   ```python
   from functools import wraps
   import hashlib
   import pickle
   from pathlib import Path

   def disk_cache(cache_dir: Path):
       """Decorator for disk-based caching."""
       cache_dir.mkdir(parents=True, exist_ok=True)

       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               key = hashlib.md5(pickle.dumps((args, kwargs))).hexdigest()
               cache_file = cache_dir / f"{func.__name__}_{key}.pkl"

               if cache_file.exists():
                   return pickle.load(open(cache_file, "rb"))

               result = func(*args, **kwargs)
               pickle.dump(result, open(cache_file, "wb"))
               return result
           return wrapper
       return decorator
   ```

2. **Vectorization**: Replace loops with pandas operations
   - Example: Replace `for ticker in tickers: fetch(ticker)`
   - With: Batch fetching + concat

3. **Parallelization** (`src/utils/market_data.py`):
   ```python
   from joblib import Parallel, delayed

   def parallel_ticker_fetch(tickers, start_date, end_date, n_jobs=-1):
       """Fetch tickers in parallel."""
       results = Parallel(n_jobs=n_jobs)(
           delayed(fetch_ticker_data)(ticker, start_date, end_date)
           for ticker in tickers
       )
       return pd.concat([r for r in results if not r.empty])
   ```

**Benchmarking**:
- Time current `fetch_bulk_data()` for 40 tickers
- Time optimized parallel version
- Target: >50% speedup

**Dependencies**: Add `joblib>=1.3.0` to requirements.txt

**Checkpoint**: Create `results/checkpoints/phase4-checkpoint-6.md`

---

### Task 4.7: Pre-commit Hooks (2-3 hours) - FINAL TASK

**Implementation**: See `PHASE4_ADVANCED_FEATURES.md` lines 652-694

**Create `.pre-commit-config.yaml`**:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: [tests/, --cov=src, --cov-fail-under=80]
```

**Install**:
```bash
pip install pre-commit
pre-commit install
```

**Test**:
```bash
# Make a change and commit
git add .
git commit -m "test: verify pre-commit hooks"
# Should run black, ruff, pytest automatically
```

**Dependencies**: Add `pre-commit>=3.5.0` to requirements.txt

**Checkpoint**: Create `results/checkpoints/phase4-final.md`

---

## 🔍 Quality Standards (Maintained from Phases 1-3)

### Code Quality

1. **Type Hints**: PEP 604 syntax (`str | None` not `Optional[str]`)
2. **Docstrings**: Google style with Args, Returns, Raises, Examples
3. **Logging**: Use `from src.utils.logger import get_logger`, no print()
4. **Exceptions**: Use custom exceptions from `src.utils.exceptions`
5. **Testing**: >80% coverage, pytest with fixtures from `tests/conftest.py`
6. **Configuration**: Load from `config/config.toml` (no hardcoded values)

### Git Workflow

1. **Branch**: Create `phase4-advanced-features` from `main`
2. **Commits**: Descriptive messages, conventional format
3. **Checkpoints**: Generate report every 8-10 hours in `results/checkpoints/`
4. **Push**: Push branch to GitHub after each checkpoint

### Checkpoint Report Template

Every 8-10 hours, create `results/checkpoints/phase4-checkpoint-N.md`:

```markdown
## Phase 4 Checkpoint [N/7]

**Time Spent**: X hours
**Completion**: XX% of phase

### Completed Tasks
- ✅ Task 4.1: Transaction costs (100%)
- ✅ Task 4.2: Advanced risk metrics (100%)

### In Progress
- 🔄 Task 4.3: Robustness framework (60% complete, ETA 4 hours)

### Blocked/Issues
- None (or describe issues)

### Test Status
- Tests passing: XX/YY
- Coverage: XX%
- CI/CD: ✅ Green

### Code Quality
- [x] Type hints (PEP 604)
- [x] Docstrings complete
- [x] Logging (no print)
- [x] Tests >80%
- [x] Config-driven (no hardcoded values)

### Performance Metrics
- Backtest speed: XX seconds (baseline YY seconds, ZZ% improvement)

### Next Steps (Next 8-10 hours)
- Complete robustness framework
- Start configuration management

### Validation
- [x] Code formatted (black)
- [x] Linting passed (ruff)
- [x] Tests passing
- [ ] Benchmarks collected
```

---

## 🚀 Getting Started (Step-by-Step)

### 1. Create Feature Branch
```bash
cd /Users/nirvanchitnis/ACCT445-Showcase
git checkout main
git pull origin main
git checkout -b phase4-advanced-features
```

### 2. Read Phase 4 Directive
```bash
cat PHASE4_ADVANCED_FEATURES.md
# Focus on Task 4.1 first (Transaction Costs)
```

### 3. Start with Task 4.1 (Transaction Costs)
```bash
# Create new file
touch src/utils/transaction_costs.py
# Implement according to PHASE4_ADVANCED_FEATURES.md lines 34-183
```

### 4. Run Tests Frequently
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 5. First Checkpoint (~10 hours)
- Complete Tasks 4.1 + 4.2 (Transaction Costs + Risk Metrics)
- Run validation:
  ```bash
  pytest tests/test_transaction_costs.py -v
  pytest tests/test_performance_metrics.py -v
  ```
- Create `results/checkpoints/phase4-checkpoint-1.md`
- Commit and push:
  ```bash
  git add .
  git commit -m "feat(trading): Implement transaction cost modeling and advanced risk metrics"
  git push -u origin phase4-advanced-features
  ```

### 6. Continue with Tasks 4.3-4.7
- Follow directive order
- Generate checkpoint every 8-10 hours
- Maintain >80% test coverage
- Keep CI/CD green

---

## 📚 Key Dependencies

### Already Installed (Phases 1-3)
- pandas >= 2.1.0
- numpy >= 1.26.0
- scipy >= 1.11.0
- statsmodels >= 0.14.0
- linearmodels >= 5.3.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- pytest >= 7.4.0
- yfinance >= 0.2.0

### Need to Add (Phase 4)
```txt
# Add to requirements.txt:
toml>=0.10.2
dvc>=3.0.0
joblib>=1.3.0
pre-commit>=3.5.0
tqdm>=4.66.0
```

Then:
```bash
pip install -r requirements.txt
```

---

## ⚠️ Known Issues from Previous Phases

### Import Paths
All imports should use absolute paths:
```python
from src.utils.logger import get_logger
from src.utils.config import get_config_value  # NEW in Phase 4
from src.analysis.decile_backtest import run_decile_backtest
```

Not relative imports:
```python
from utils.logger import get_logger  # ❌ Wrong
from ..utils.logger import get_logger  # ❌ Wrong
```

### Configuration Migration
When adding config.toml, update existing code incrementally:
1. Add config loading at top of modules
2. Replace hardcoded values one at a time
3. Test after each change
4. Don't break existing functionality

---

## 📊 Phase 4 Success Definition

Phase 4 is **complete** when:

1. ✅ All 7 tasks complete (4.1-4.7)
2. ✅ Test coverage >80% for each new module
3. ✅ All tests passing (0 failures)
4. ✅ CI/CD pipeline green
5. ✅ All checkpoints generated
6. ✅ Backtest speed improved >50% (benchmarked)
7. ✅ Transaction costs realistic (2-5 bps per trade)
8. ✅ Robustness checks confirm Phase 3 findings
9. ✅ All parameters configurable via TOML
10. ✅ DVC tracking data files
11. ✅ Pre-commit hooks installed and working
12. ✅ Final checkpoint report shows 100% completion
13. ✅ Branch pushed to GitHub
14. ✅ Ready for Phase 5 (create PR but don't merge yet)

---

## 🎯 What Comes After Phase 4

**Phase 5**: Production Deployment (30-40 hours)
- Docker containerization
- API endpoints (FastAPI)
- Streamlit dashboard
- Documentation (Sphinx)
- Production monitoring
- Deployment guide

**Dependencies**: Phase 4 modules (transaction costs, robustness, config, optimization) must be complete and tested.

---

## 📞 Questions / Blockers

If you encounter issues:

1. **Check Phase 1-3 code**: Look at existing modules for patterns
2. **Review existing tests**: See how similar functionality is tested
3. **Follow directive exactly**: `PHASE4_ADVANCED_FEATURES.md` has detailed implementations
4. **Log, don't halt**: If something fails, log error and continue with other tasks
5. **Ask for clarification**: If directive is unclear, note in checkpoint and proceed with best judgment

---

## ✅ Pre-Flight Checklist

Before starting Phase 4, verify:

- [x] Phase 3 merged to main
- [x] Working directory clean (`git status`)
- [x] All Phase 3 tests passing (103 tests, 88.78% coverage)
- [x] `PHASE4_ADVANCED_FEATURES.md` read and understood
- [x] `requirements.txt` up to date
- [x] Ready to create `phase4-advanced-features` branch

---

**Ready to Start**: YES ✅

**First Task**: Task 4.1 - Implement `src/utils/transaction_costs.py`

**Estimated First Checkpoint**: 8-10 hours (Tasks 4.1 + 4.2 complete)

**Good luck! Remember to generate checkpoint reports every 8-10 hours.**

---

**Document Control**

**Version**: 1.0
**Created**: 2025-11-08
**Author**: Claude Code
**For**: Codex
**Phase**: 4 of 5
**Status**: Active handoff
