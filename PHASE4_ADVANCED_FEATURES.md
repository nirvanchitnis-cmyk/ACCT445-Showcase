# Phase 4: Advanced Features & Production Readiness

**Phase**: 4 of 5
**Estimated Time**: 35-45 hours
**Dependencies**: Phase 3 complete (real data integration working)
**Status**: 🔴 Blocked (requires Phase 3)

---

## 🎯 Objectives

Transform from research tool to production trading system:
1. Transaction cost modeling (realistic bid-ask, market impact, trading fees)
2. Advanced risk metrics (VaR, CVaR, tail risk, factor exposures)
3. Robustness checks (bootstrap, permutation tests, subsamples)
4. Configuration management (centralized parameters)
5. Data versioning (track CNOI updates with DVC)
6. Performance optimization (caching, vectorization, parallelization)
7. Code quality automation (pre-commit hooks)

**Success Criteria**:
- ✅ Transaction costs reduce backtest returns by realistic 2-5 bps per trade
- ✅ Advanced risk metrics match industry standards (validated against QuantLib)
- ✅ Robustness checks confirm main results hold
- ✅ All parameters configurable via TOML file
- ✅ Data versioned and tracked
- ✅ Backtest speed improved >50%
- ✅ Pre-commit hooks enforce quality

---

## 📋 Task Breakdown

### Task 4.1: Transaction Cost Modeling (8-10 hours)

#### 4.1.1: Create `src/utils/transaction_costs.py`

```python
"""
Transaction cost modeling for backtests.

Components:
1. Bid-ask spread (function of volatility, liquidity)
2. Market impact (function of trade size, ADV)
3. Fixed trading fees (commission)
4. Slippage
"""

import pandas as pd
import numpy as np
from typing import Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)


def estimate_bid_ask_spread(
    volatility: pd.Series,
    market_cap: pd.Series,
    base_spread_bps: float = 5.0
) -> pd.Series:
    """
    Estimate bid-ask spread as function of volatility and size.

    Args:
        volatility: Annualized volatility
        market_cap: Market capitalization
        base_spread_bps: Base spread in basis points

    Returns:
        Estimated spread in basis points

    Formula:
        spread = base_spread * (1 + volatility) * (1 / sqrt(market_cap_billions))
    """
    mcap_billions = market_cap / 1e9
    spread = base_spread_bps * (1 + volatility) * (1 / np.sqrt(mcap_billions.clip(lower=1)))
    return spread.clip(lower=1.0, upper=50.0)  # Reasonable bounds


def compute_market_impact(
    trade_value: float,
    avg_daily_volume: float,
    impact_coefficient: float = 0.1
) -> float:
    """
    Compute market impact cost.

    Args:
        trade_value: Dollar value of trade
        avg_daily_volume: Average daily trading volume (dollars)
        impact_coefficient: Impact coefficient (default 0.1)

    Returns:
        Market impact cost in basis points

    Formula (Almgren-Chriss):
        impact = impact_coef * (trade_value / ADV)^0.5 * volatility
    """
    if avg_daily_volume <= 0:
        return 100.0  # High cost for illiquid

    participation_rate = trade_value / avg_daily_volume
    impact_bps = impact_coefficient * np.sqrt(participation_rate) * 10000

    return min(impact_bps, 100.0)  # Cap at 100 bps


def apply_transaction_costs(
    backtest_returns: pd.DataFrame,
    turnover: float,
    avg_spread_bps: float = 5.0,
    commission_bps: float = 1.0,
    avg_impact_bps: float = 2.0
) -> pd.DataFrame:
    """
    Apply transaction costs to backtest returns.

    Args:
        backtest_returns: DataFrame with gross returns
        turnover: Portfolio turnover rate (1.0 = 100% turnover per period)
        avg_spread_bps: Average bid-ask spread
        commission_bps: Commission rate
        avg_impact_bps: Average market impact

    Returns:
        DataFrame with net returns (after costs)

    Example:
        >>> net_returns = apply_transaction_costs(
        >>>     gross_returns,
        >>>     turnover=0.5,  # 50% turnover per quarter
        >>>     avg_spread_bps=5.0
        >>> )
    """
    # Total cost per trade (one-way)
    total_cost_bps = (avg_spread_bps / 2) + commission_bps + avg_impact_bps

    # Cost per period = turnover * one-way cost
    cost_per_period = turnover * total_cost_bps / 10000

    # Apply to returns
    result = backtest_returns.copy()
    result["gross_ret"] = result["ret"]
    result["transaction_cost"] = cost_per_period
    result["net_ret"] = result["gross_ret"] - cost_per_period

    logger.info(f"Transaction costs applied: {cost_per_period:.4%} per period")

    return result


if __name__ == "__main__":
    # Demo
    np.random.seed(42)

    # Simulate backtest returns
    n = 40  # Quarterly rebalancing over 10 years
    gross_returns = pd.DataFrame({
        "quarter": pd.period_range("2015Q1", periods=n, freq="Q"),
        "ret": np.random.normal(0.03, 0.05, n)  # 3% mean, 5% std per quarter
    })

    # Apply costs
    net_returns = apply_transaction_costs(
        gross_returns,
        turnover=0.5,  # 50% quarterly turnover
        avg_spread_bps=5.0,
        commission_bps=1.0,
        avg_impact_bps=2.0
    )

    print("=== TRANSACTION COST IMPACT ===")
    print(f"Gross return (annualized): {net_returns['gross_ret'].mean() * 4:.2%}")
    print(f"Transaction cost (annualized): {net_returns['transaction_cost'].mean() * 4:.2%}")
    print(f"Net return (annualized): {net_returns['net_ret'].mean() * 4:.2%}")
    print(f"Cost drag: {(net_returns['gross_ret'].mean() - net_returns['net_ret'].mean()) / net_returns['gross_ret'].mean():.1%}")
```

**Tests**: `tests/test_transaction_costs.py`

**Checkpoint 4.1**: Transaction costs implemented, integrated into notebooks

---

### Task 4.2: Advanced Risk Metrics (8-10 hours)

#### 4.2.1: Extend `src/utils/performance_metrics.py`

Add functions:
- `value_at_risk(returns, confidence=0.95)`: Historical VaR
- `conditional_var(returns, confidence=0.95)`: CVaR / Expected Shortfall
- `tail_ratio(returns)`: Right tail / left tail
- `skewness(returns)`: Return skewness
- `kurtosis(returns)`: Return kurtosis (excess)
- `downside_capture(returns, benchmark)`: Downside capture ratio
- `upside_capture(returns, benchmark)`: Upside capture ratio
- `omega_ratio(returns, threshold=0)`: Omega ratio
- `rolling_volatility(returns, window=21)`: Rolling vol for vol targeting

**Example**:

```python
def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Compute Value at Risk.

    Args:
        returns: Return series
        confidence: Confidence level (0.95 = 95%)
        method: "historical" or "parametric"

    Returns:
        VaR (negative value representing loss)

    Example:
        >>> var_95 = value_at_risk(daily_returns, confidence=0.95)
        >>> print(f"95% VaR: {var_95:.2%}")  # e.g., -2.5%
    """
    if method == "historical":
        return np.percentile(returns, (1 - confidence) * 100)
    elif method == "parametric":
        from scipy.stats import norm
        return norm.ppf(1 - confidence, loc=returns.mean(), scale=returns.std())
    else:
        raise ValueError("method must be 'historical' or 'parametric'")


def conditional_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Compute Conditional Value at Risk (Expected Shortfall).

    CVaR is the expected loss given that VaR has been exceeded.

    Args:
        returns: Return series
        confidence: Confidence level

    Returns:
        CVaR (negative value)
    """
    var = value_at_risk(returns, confidence, method="historical")
    return returns[returns <= var].mean()
```

**Checkpoint 4.2**: Advanced risk metrics implemented and validated

---

### Task 4.3: Robustness Framework (8-10 hours)

#### 4.3.1: Create `src/analysis/robustness.py`

```python
"""
Robustness checks for backtest results.

Methods:
1. Bootstrap: Resample returns to test stability
2. Permutation test: Shuffle labels to test significance
3. Subsample analysis: Split by time period, market regime
4. Monte Carlo: Simulate alternative scenarios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable
from tqdm import tqdm
from src.analysis.decile_backtest import run_decile_backtest
from src.utils.logger import get_logger

logger = get_logger(__name__)


def bootstrap_backtest(
    df: pd.DataFrame,
    score_col: str,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42
) -> Dict:
    """
    Bootstrap resampling of backtest results.

    Args:
        df: Input data
        score_col: Score column (e.g., "CNOI")
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level for intervals
        random_seed: Random seed

    Returns:
        Dictionary with:
            - mean: Bootstrap mean of long-short spread
            - ci_lower: Lower confidence interval
            - ci_upper: Upper confidence interval
            - distribution: Full bootstrap distribution

    Example:
        >>> boot_results = bootstrap_backtest(df, score_col="CNOI", n_bootstrap=1000)
        >>> print(f"95% CI: [{boot_results['ci_lower']:.4f}, {boot_results['ci_upper']:.4f}]")
    """
    np.random.seed(random_seed)
    long_short_spreads = []

    logger.info(f"Running {n_bootstrap} bootstrap samples...")

    for i in tqdm(range(n_bootstrap)):
        # Resample with replacement
        sample = df.sample(n=len(df), replace=True)

        # Run backtest on sample
        try:
            summary, long_short = run_decile_backtest(sample, score_col=score_col, n_deciles=10)
            long_short_spreads.append(long_short["mean_ret"])
        except:
            pass  # Skip failed samples

    long_short_spreads = np.array(long_short_spreads)

    # Compute confidence intervals
    alpha = 1 - confidence
    ci_lower = np.percentile(long_short_spreads, alpha / 2 * 100)
    ci_upper = np.percentile(long_short_spreads, (1 - alpha / 2) * 100)

    logger.info(f"Bootstrap complete: mean = {long_short_spreads.mean():.4f}, "
                f"CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

    return {
        "mean": long_short_spreads.mean(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "distribution": long_short_spreads,
    }


def permutation_test(
    df: pd.DataFrame,
    score_col: str,
    n_permutations: int = 1000,
    random_seed: int = 42
) -> Dict:
    """
    Permutation test: Shuffle score labels to test significance.

    Null hypothesis: No relationship between score and returns.

    Args:
        df: Input data
        score_col: Score column to permute
        n_permutations: Number of permutations
        random_seed: Random seed

    Returns:
        Dictionary with:
            - observed: Observed long-short spread
            - p_value: Permutation p-value
            - null_distribution: Distribution under null

    Example:
        >>> perm_results = permutation_test(df, score_col="CNOI", n_permutations=1000)
        >>> print(f"Permutation p-value: {perm_results['p_value']:.4f}")
    """
    np.random.seed(random_seed)

    # Observed statistic
    summary, long_short_obs = run_decile_backtest(df, score_col=score_col, n_deciles=10)
    observed = long_short_obs["mean_ret"]

    logger.info(f"Observed long-short spread: {observed:.4f}")
    logger.info(f"Running {n_permutations} permutations...")

    # Null distribution
    null_spreads = []

    for i in tqdm(range(n_permutations)):
        df_permuted = df.copy()
        df_permuted[score_col] = np.random.permutation(df_permuted[score_col].values)

        try:
            summary, long_short = run_decile_backtest(df_permuted, score_col=score_col, n_deciles=10)
            null_spreads.append(long_short["mean_ret"])
        except:
            pass

    null_spreads = np.array(null_spreads)

    # P-value: fraction of permutations with spread >= observed
    p_value = (np.abs(null_spreads) >= np.abs(observed)).mean()

    logger.info(f"Permutation test complete: p-value = {p_value:.4f}")

    return {
        "observed": observed,
        "p_value": p_value,
        "null_distribution": null_spreads,
    }


def subsample_analysis(
    df: pd.DataFrame,
    score_col: str,
    split_col: str,
    split_values: List
) -> pd.DataFrame:
    """
    Run backtest on multiple subsamples.

    Args:
        df: Input data
        score_col: Score column
        split_col: Column to split on (e.g., "year", "market_regime")
        split_values: List of values to split on

    Returns:
        DataFrame with results for each subsample

    Example:
        >>> # Test by year
        >>> subsample_results = subsample_analysis(
        >>>     df,
        >>>     score_col="CNOI",
        >>>     split_col="year",
        >>>     split_values=[2020, 2021, 2022, 2023]
        >>> )
    """
    results = []

    for value in split_values:
        subset = df[df[split_col] == value]

        if len(subset) < 100:
            logger.warning(f"Skipping {split_col}={value}: only {len(subset)} observations")
            continue

        logger.info(f"Running backtest for {split_col}={value} ({len(subset)} obs)")

        summary, long_short = run_decile_backtest(subset, score_col=score_col, n_deciles=10)

        results.append({
            split_col: value,
            "n_obs": len(subset),
            "long_short_ret": long_short["mean_ret"],
            "t_stat": long_short["t_stat"],
            "p_value": long_short["p_value"],
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Demo
    np.random.seed(42)

    # Simulate data with signal
    n = 1000
    df = pd.DataFrame({
        "CNOI": np.random.uniform(10, 30, n),
        "ret_fwd": np.random.normal(0, 0.02, n),
        "quarter": np.random.choice(pd.period_range("2020Q1", periods=16, freq="Q"), n),
    })

    # Add signal: CNOI predicts returns
    df["ret_fwd"] = df["ret_fwd"] - 0.002 * (df["CNOI"] - 20)

    # Bootstrap
    boot = bootstrap_backtest(df, score_col="CNOI", n_bootstrap=100)
    print(f"Bootstrap 95% CI: [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}]")

    # Permutation
    perm = permutation_test(df, score_col="CNOI", n_permutations=100)
    print(f"Permutation p-value: {perm['p_value']:.4f}")
```

**Checkpoint 4.3**: Robustness framework implemented

---

### Task 4.4: Configuration Management (4-5 hours)

#### 4.4.1: Create `config/config.toml`

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
rebalance_frequency = "Q"  # Quarterly

[transaction_costs]
avg_spread_bps = 5.0
commission_bps = 1.0
avg_impact_bps = 2.0
assumed_turnover = 0.5  # 50% per rebalance

[risk_metrics]
var_confidence = 0.95
periods_per_year = 252
risk_free_rate = 0.03  # 3% annual

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

#### 4.4.2: Create `src/utils/config.py`

```python
"""
Configuration management using TOML files.
"""

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

**Checkpoint 4.4**: All parameters moved to config.toml

---

### Task 4.5: Data Versioning with DVC (4-5 hours)

#### 4.5.1: Install and configure DVC

```bash
pip install dvc
dvc init
dvc add config/sample_cnoi.csv
dvc add data/cache/
git add config/.gitignore config/sample_cnoi.csv.dvc data/cache.dvc
git commit -m "Add DVC tracking for data files"
```

#### 4.5.2: Document DVC usage in README

**Checkpoint 4.5**: DVC tracking data files

---

### Task 4.6: Performance Optimization (6-8 hours)

#### 4.6.1: Caching decorator

```python
from functools import lru_cache, wraps
import hashlib
import pickle
from pathlib import Path


def disk_cache(cache_dir: Path):
    """Decorator for disk-based caching."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args
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

#### 4.6.2: Vectorization

Replace loops with pandas operations where possible.

#### 4.6.3: Parallelization

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

**Checkpoint 4.6**: Backtest speed improved >50%

---

### Task 4.7: Pre-commit Hooks (2-3 hours)

#### 4.7.1: Create `.pre-commit-config.yaml`

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

```bash
pip install pre-commit
pre-commit install
```

**Checkpoint 4.7**: Pre-commit hooks enforcing quality

---

## 📊 Definition of Done (Phase 4)

### Features Complete
- [x] Transaction costs implemented and realistic
- [x] Advanced risk metrics (VaR, CVaR, tail risk, etc.)
- [x] Robustness framework (bootstrap, permutation, subsample)
- [x] Configuration management via TOML
- [x] DVC tracking data files
- [x] Performance optimizations (caching, vectorization, parallelization)
- [x] Pre-commit hooks enforcing quality

### Testing
- [x] All new features have >80% test coverage
- [x] Integration tests confirm features work together
- [x] Performance benchmarks show >50% speedup

### Documentation
- [x] README updated with advanced features
- [x] Config file documented
- [x] DVC usage explained

### Validation
- [x] Transaction costs match literature (2-5 bps)
- [x] Risk metrics validated against QuantLib
- [x] Robustness checks confirm main results

---

## ✅ Ready for Phase 5 When...

1. ✅ All advanced features working
2. ✅ Config-driven analysis reproducible
3. ✅ Performance benchmarks met
4. ✅ CI/CD green with all checks
5. ✅ Pre-commit enforcing quality

**Next Phase**: `PHASE5_PRODUCTION_DEPLOYMENT.md`

---

**Document Control**

**Version**: 1.0
**Status**: 🔴 Blocked (requires Phase 3)
**Estimated Time**: 35-45 hours
**Dependencies**: Phase 3 (real data integration)
**Next Phase**: Phase 5 (Production Deployment)
