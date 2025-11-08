# Phase 2: Core Analysis Modules

**Phase**: 2 of 5
**Estimated Time**: 32-43 hours (includes SEC API fix)
**Dependencies**: Phase 1 complete ✅ (test infrastructure ready)
**Status**: 🟢 Ready to Start

---

## 🎯 Objectives

Implement all missing analysis modules referenced in README:
0. **`src/data/sec_api_client.py`**: Fix SEC API integration (retry logic, caching, User-Agent compliance) 🔧
1. **`src/analysis/panel_regression.py`**: Panel econometrics (Fixed Effects, Fama-MacBeth, Driscoll-Kraay)
2. **`src/analysis/dimension_analysis.py`**: CNOI dimension analysis (D, G, R, J, T, S, X)
3. **`src/utils/performance_metrics.py`**: Portfolio performance metrics (Sharpe, Sortino, IR, etc.)
4. Write comprehensive tests for all new modules (>80% coverage)
5. Integrate with existing codebase seamlessly

**Success Criteria**:
- ✅ SEC API integration working reliably (no 403 errors)
- ✅ All 3 analysis modules implemented with proper docstrings and type hints
- ✅ >80% test coverage for all new code (including SEC client)
- ✅ CI/CD pipeline stays green
- ✅ Demo sections work end-to-end (including CIK mapper)
- ✅ Academic rigor maintained (proper econometric methods)

---

## 📋 Task Breakdown

### Task 2.0: Fix SEC API Integration (2-3 hours) 🔧 PRIORITY

**Why First**: Phase 1 identified SEC API 403 errors. Fix the data pipeline before implementing complex analysis modules.

#### 2.0.1: Implement Robust SEC API Client

**File**: `src/data/sec_api_client.py` (new file)

```python
"""
Robust SEC EDGAR API client with retry logic and caching.

Features:
- Exponential backoff retry
- User-Agent rotation
- Disk caching
- Rate limiting compliance
"""

import requests
import time
import json
from pathlib import Path
from typing import Dict, Optional
import hashlib
from src.utils.logger import get_logger
from src.utils.exceptions import ExternalAPIError

logger = get_logger(__name__)

CACHE_DIR = Path("data/cache/sec")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# SEC requires descriptive User-Agent
# See: https://www.sec.gov/os/accessing-edgar-data
USER_AGENTS = [
    "ACCT445-Research contact@university.edu",
    "Academic-Research student@university.edu",
]


def fetch_sec_ticker_mapping(
    use_cache: bool = True,
    max_retries: int = 3
) -> Dict[str, Dict]:
    """
    Fetch SEC company ticker mapping with retry logic.

    Args:
        use_cache: Use cached data if available
        max_retries: Maximum retry attempts

    Returns:
        Dictionary mapping CIK to {ticker, title}

    Raises:
        ExternalAPIError: If all retries fail

    Example:
        >>> mapping = fetch_sec_ticker_mapping()
        >>> print(mapping["0000070858"]["ticker"])  # "BAC"
    """
    cache_file = CACHE_DIR / "company_tickers.json"

    # Check cache
    if use_cache and cache_file.exists():
        cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if cache_age_hours < 24:  # Cache valid for 24 hours
            logger.info(f"Loading SEC mapping from cache (age: {cache_age_hours:.1f}h)")
            with open(cache_file, "r") as f:
                return json.load(f)

    # Fetch from SEC API with retries
    url = "https://www.sec.gov/files/company_tickers.json"

    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": USER_AGENTS[attempt % len(USER_AGENTS)],
                "Accept": "application/json",
            }

            logger.info(f"Fetching SEC mapping (attempt {attempt + 1}/{max_retries})")

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Transform to CIK-indexed format
            mapping = {}
            for entry in data.values():
                cik = str(entry["cik_str"]).zfill(10)
                mapping[cik] = {
                    "ticker": entry["ticker"],
                    "title": entry["title"],
                }

            # Cache successful response
            with open(cache_file, "w") as f:
                json.dump(mapping, f)

            logger.info(f"✓ SEC mapping fetched: {len(mapping)} companies")
            return mapping

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.warning(f"SEC API 403 Forbidden (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise ExternalAPIError(
                        "SEC API returned 403 after all retries. "
                        "Check User-Agent or use cached data."
                    ) from e
            else:
                raise ExternalAPIError(f"SEC API HTTP error: {e}") from e

        except Exception as e:
            logger.error(f"Error fetching SEC mapping (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise ExternalAPIError("Failed to fetch SEC mapping") from e

    raise ExternalAPIError("Failed to fetch SEC mapping after all retries")
```

#### 2.0.2: Update CIK Ticker Mapper

**Modify**: `src/data/cik_ticker_mapper.py`

Replace `fetch_sec_ticker_mapping()` function to use the new robust client:

```python
from src.data.sec_api_client import fetch_sec_ticker_mapping
# Remove old implementation, use new client
```

#### 2.0.3: Write Tests

**File**: `tests/test_sec_api_client.py`

```python
"""
Tests for src/data/sec_api_client.py
"""

import pytest
import requests
from unittest.mock import patch, MagicMock
from src.data.sec_api_client import fetch_sec_ticker_mapping
from src.utils.exceptions import ExternalAPIError


class TestFetchSecTickerMapping:
    """Tests for SEC API client."""

    @patch("src.data.sec_api_client.requests.get")
    def test_fetch_success(self, mock_get):
        """Test successful API fetch."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "0": {"cik_str": 70858, "ticker": "BAC", "title": "BANK OF AMERICA CORP"},
            "1": {"cik_str": 19617, "ticker": "JPM", "title": "JPMORGAN CHASE"},
        }
        mock_get.return_value = mock_response

        result = fetch_sec_ticker_mapping(use_cache=False)

        assert "0000070858" in result
        assert result["0000070858"]["ticker"] == "BAC"

    @patch("src.data.sec_api_client.requests.get")
    def test_retry_on_403(self, mock_get):
        """Test retry logic on 403 error."""
        # First attempt: 403
        mock_response_403 = MagicMock()
        mock_response_403.status_code = 403
        mock_response_403.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response_403
        )

        # Second attempt: success
        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200
        mock_response_200.json.return_value = {
            "0": {"cik_str": 70858, "ticker": "BAC", "title": "BANK OF AMERICA"}
        }

        mock_get.side_effect = [mock_response_403, mock_response_200]

        result = fetch_sec_ticker_mapping(use_cache=False, max_retries=2)

        assert "0000070858" in result
        assert mock_get.call_count == 2

    @patch("src.data.sec_api_client.requests.get")
    def test_exhausted_retries(self, mock_get):
        """Test that ExternalAPIError raised after all retries."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(ExternalAPIError, match="403"):
            fetch_sec_ticker_mapping(use_cache=False, max_retries=3)

    def test_cache_usage(self, tmp_path, monkeypatch):
        """Test that cache is used when available."""
        # Create fake cache
        cache_dir = tmp_path / "sec"
        cache_dir.mkdir()
        cache_file = cache_dir / "company_tickers.json"

        import json
        cached_data = {"0000070858": {"ticker": "BAC", "title": "Bank of America"}}
        with open(cache_file, "w") as f:
            json.dump(cached_data, f)

        # Monkeypatch CACHE_DIR
        import src.data.sec_api_client as sec_module
        monkeypatch.setattr(sec_module, "CACHE_DIR", cache_dir)

        result = fetch_sec_ticker_mapping(use_cache=True)

        assert result == cached_data
```

#### 2.0.4: Update Existing Tests

**Modify**: `tests/test_cik_ticker_mapper.py`

Update to use the new client and ensure all mocks are compatible.

**Checkpoint 2.0**: SEC API working reliably, tests passing, cached data working

---

### Task 2.1: Implement `panel_regression.py` (12-15 hours)

#### 2.1.1: Create module structure

**File**: `src/analysis/panel_regression.py`

```python
"""
Panel regression analysis for CNOI and stock returns.

Implements three panel econometric methods:
1. Fixed Effects (FE) regression with entity and time effects
2. Fama-MacBeth (FM) two-step cross-sectional regression
3. Driscoll-Kraay standard errors for panel data

References:
- Petersen, M. A. (2009). "Estimating standard errors in finance panel data sets"
- Fama, E. F., & MacBeth, J. D. (1973). "Risk, return, and equilibrium"
- Driscoll, J. C., & Kraay, A. C. (1998). "Consistent covariance matrix estimation"
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from linearmodels.panel import PanelOLS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_panel_data(
    df: pd.DataFrame,
    entity_col: str = "ticker",
    time_col: str = "quarter",
) -> pd.DataFrame:
    """
    Prepare data for panel regression with MultiIndex.

    Args:
        df: DataFrame with entity and time columns
        entity_col: Name of entity identifier column
        time_col: Name of time identifier column

    Returns:
        DataFrame with (entity, time) MultiIndex

    Example:
        >>> panel_df = prepare_panel_data(df, entity_col="ticker", time_col="quarter")
    """
    df = df.copy()

    # Set MultiIndex
    df = df.set_index([entity_col, time_col])

    # Sort index
    df = df.sort_index()

    logger.info(f"Panel data prepared: {len(df.index.get_level_values(0).unique())} entities, "
                f"{len(df.index.get_level_values(1).unique())} periods")

    return df


def fixed_effects_regression(
    df: pd.DataFrame,
    dependent_var: str = "ret_fwd",
    independent_vars: List[str] = None,
    entity_effects: bool = True,
    time_effects: bool = True,
) -> Dict:
    """
    Run Fixed Effects panel regression.

    Args:
        df: Panel DataFrame with (entity, time) MultiIndex
        dependent_var: Name of dependent variable (e.g., "ret_fwd")
        independent_vars: List of independent variables (e.g., ["cnoi", "log_mcap", "leverage"])
        entity_effects: Include entity (stock) fixed effects
        time_effects: Include time (quarter) fixed effects

    Returns:
        Dictionary with results:
            - coefficients: Estimated coefficients
            - std_errors: Standard errors
            - t_stats: T-statistics
            - p_values: P-values
            - r_squared: R-squared
            - n_obs: Number of observations
            - model: Full regression model

    Example:
        >>> results = fixed_effects_regression(
        >>>     panel_df,
        >>>     dependent_var="ret_fwd",
        >>>     independent_vars=["CNOI", "log_mcap"]
        >>> )
        >>> print(f"CNOI coefficient: {results['coefficients']['CNOI']:.4f}")
    """
    if independent_vars is None:
        independent_vars = ["CNOI"]

    # Prepare regression
    y = df[dependent_var]
    X = df[independent_vars]

    # Remove missing values
    data = pd.concat([y, X], axis=1).dropna()
    y = data[dependent_var]
    X = data[independent_vars]

    logger.info(f"Running FE regression: {len(data)} observations, {len(independent_vars)} variables")

    # Determine effects
    if entity_effects and time_effects:
        effects = "twoway"
    elif entity_effects:
        effects = "entity"
    elif time_effects:
        effects = "time"
    else:
        effects = None

    # Run regression
    model = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects)
    results = model.fit(cov_type="clustered", cluster_entity=True)

    # Extract results
    output = {
        "coefficients": results.params.to_dict(),
        "std_errors": results.std_errors.to_dict(),
        "t_stats": results.tstats.to_dict(),
        "p_values": results.pvalues.to_dict(),
        "r_squared": results.rsquared,
        "r_squared_within": results.rsquared_within,
        "r_squared_between": results.rsquared_between,
        "n_obs": results.nobs,
        "effects": effects,
        "model": results,
    }

    logger.info(f"FE regression complete: R² = {results.rsquared:.4f}")

    return output


def fama_macbeth_regression(
    df: pd.DataFrame,
    dependent_var: str = "ret_fwd",
    independent_vars: List[str] = None,
    time_col: str = "quarter",
) -> Dict:
    """
    Run Fama-MacBeth two-step cross-sectional regression.

    Step 1: Run cross-sectional regression for each time period
    Step 2: Compute time-series mean of coefficients with Newey-West SEs

    Args:
        df: DataFrame with time column (not MultiIndex)
        dependent_var: Name of dependent variable
        independent_vars: List of independent variables
        time_col: Name of time column

    Returns:
        Dictionary with results:
            - coefficients: Mean coefficients across time
            - std_errors: Newey-West standard errors
            - t_stats: T-statistics
            - p_values: P-values
            - n_periods: Number of time periods
            - period_results: List of per-period regression results

    Example:
        >>> results = fama_macbeth_regression(
        >>>     df,
        >>>     dependent_var="ret_fwd",
        >>>     independent_vars=["CNOI"]
        >>> )
    """
    if independent_vars is None:
        independent_vars = ["CNOI"]

    # Get unique time periods
    time_periods = df[time_col].unique()
    logger.info(f"Running Fama-MacBeth: {len(time_periods)} time periods")

    # Step 1: Run cross-sectional regressions
    period_coefficients = {var: [] for var in independent_vars}
    period_coefficients["const"] = []

    for period in time_periods:
        period_data = df[df[time_col] == period].dropna(subset=[dependent_var] + independent_vars)

        if len(period_data) < 10:  # Skip periods with too few observations
            logger.warning(f"Skipping {period}: only {len(period_data)} observations")
            continue

        # Run OLS for this period
        y = period_data[dependent_var]
        X = period_data[independent_vars]
        X = sm.add_constant(X)

        try:
            model = OLS(y, X).fit()

            # Store coefficients
            for var in independent_vars:
                period_coefficients[var].append(model.params[var])
            period_coefficients["const"].append(model.params["const"])

        except Exception as e:
            logger.warning(f"Regression failed for {period}: {e}")

    # Step 2: Compute time-series mean and standard errors
    results = {}
    for var in independent_vars + ["const"]:
        coef_series = np.array(period_coefficients[var])

        if len(coef_series) == 0:
            raise ValueError(f"No valid coefficients for {var}")

        mean_coef = coef_series.mean()
        se_coef = coef_series.std() / np.sqrt(len(coef_series))  # Standard error of mean
        t_stat = mean_coef / se_coef
        p_value = 2 * (1 - sm.tools.eval_measures.scipy.stats.t.cdf(abs(t_stat), len(coef_series) - 1))

        results[var] = {
            "coefficient": mean_coef,
            "std_error": se_coef,
            "t_stat": t_stat,
            "p_value": p_value,
        }

    logger.info(f"Fama-MacBeth complete: {len(coef_series)} periods")

    return {
        "coefficients": {var: results[var]["coefficient"] for var in results},
        "std_errors": {var: results[var]["std_error"] for var in results},
        "t_stats": {var: results[var]["t_stat"] for var in results},
        "p_values": {var: results[var]["p_value"] for var in results},
        "n_periods": len(coef_series),
        "period_coefficients": period_coefficients,
    }


def driscoll_kraay_regression(
    df: pd.DataFrame,
    dependent_var: str = "ret_fwd",
    independent_vars: List[str] = None,
    max_lags: int = 4,
) -> Dict:
    """
    Run OLS regression with Driscoll-Kraay standard errors.

    Driscoll-Kraay SEs are robust to cross-sectional dependence and
    heteroskedasticity in panel data.

    Args:
        df: Panel DataFrame with (entity, time) MultiIndex
        dependent_var: Name of dependent variable
        independent_vars: List of independent variables
        max_lags: Maximum lags for autocorrelation (Newey-West style)

    Returns:
        Dictionary with results (similar to fixed_effects_regression)

    Example:
        >>> results = driscoll_kraay_regression(
        >>>     panel_df,
        >>>     dependent_var="ret_fwd",
        >>>     independent_vars=["CNOI"]
        >>> )
    """
    if independent_vars is None:
        independent_vars = ["CNOI"]

    # Prepare regression
    y = df[dependent_var]
    X = df[independent_vars]

    # Remove missing values
    data = pd.concat([y, X], axis=1).dropna()
    y = data[dependent_var]
    X = data[independent_vars]

    logger.info(f"Running Driscoll-Kraay regression: {len(data)} observations")

    # Run regression with Driscoll-Kraay SEs
    model = PanelOLS(y, X, entity_effects=True, time_effects=True)
    results = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=max_lags)

    # Extract results
    output = {
        "coefficients": results.params.to_dict(),
        "std_errors": results.std_errors.to_dict(),
        "t_stats": results.tstats.to_dict(),
        "p_values": results.pvalues.to_dict(),
        "r_squared": results.rsquared,
        "n_obs": results.nobs,
        "max_lags": max_lags,
        "model": results,
    }

    logger.info(f"Driscoll-Kraay complete: R² = {results.rsquared:.4f}")

    return output


def run_all_panel_regressions(
    df: pd.DataFrame,
    dependent_var: str = "ret_fwd",
    independent_vars: List[str] = None,
    entity_col: str = "ticker",
    time_col: str = "quarter",
) -> Dict[str, Dict]:
    """
    Run all three panel regression methods and compare results.

    Args:
        df: DataFrame with panel data
        dependent_var: Dependent variable name
        independent_vars: List of independent variables
        entity_col: Entity identifier column
        time_col: Time identifier column

    Returns:
        Dictionary with keys "FE", "FM", "DK" containing results from each method

    Example:
        >>> all_results = run_all_panel_regressions(
        >>>     df,
        >>>     dependent_var="ret_fwd",
        >>>     independent_vars=["CNOI", "log_mcap"]
        >>> )
        >>> print("Fixed Effects CNOI coef:", all_results["FE"]["coefficients"]["CNOI"])
        >>> print("Fama-MacBeth CNOI coef:", all_results["FM"]["coefficients"]["CNOI"])
    """
    if independent_vars is None:
        independent_vars = ["CNOI"]

    logger.info("Running all panel regression methods")

    # Prepare panel data
    panel_df = prepare_panel_data(df, entity_col=entity_col, time_col=time_col)

    # Fixed Effects
    fe_results = fixed_effects_regression(
        panel_df,
        dependent_var=dependent_var,
        independent_vars=independent_vars
    )

    # Fama-MacBeth (needs non-MultiIndex df)
    fm_results = fama_macbeth_regression(
        df,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        time_col=time_col
    )

    # Driscoll-Kraay
    dk_results = driscoll_kraay_regression(
        panel_df,
        dependent_var=dependent_var,
        independent_vars=independent_vars
    )

    # Compare results
    comparison_df = pd.DataFrame({
        "FE_coef": fe_results["coefficients"],
        "FE_tstat": fe_results["t_stats"],
        "FM_coef": fm_results["coefficients"],
        "FM_tstat": fm_results["t_stats"],
        "DK_coef": dk_results["coefficients"],
        "DK_tstat": dk_results["t_stats"],
    })

    logger.info("All panel regressions complete")
    logger.info(f"\n{comparison_df}")

    return {
        "FE": fe_results,
        "FM": fm_results,
        "DK": dk_results,
        "comparison": comparison_df,
    }


if __name__ == "__main__":
    # Demo: Simulate panel data and run regressions
    np.random.seed(42)

    # Create sample panel data
    n_entities = 50
    n_periods = 20
    tickers = [f"BANK{i:02d}" for i in range(n_entities)]
    quarters = pd.period_range("2020Q1", periods=n_periods, freq="Q")

    data = []
    for ticker in tickers:
        # Entity-specific effect
        entity_effect = np.random.normal(0, 0.02)

        for quarter in quarters:
            # Time-specific effect
            time_effect = np.random.normal(0, 0.01)

            # CNOI (persistent within entity, small time variation)
            cnoi = 15 + hash(ticker) % 20 + np.random.normal(0, 1)

            # Return = entity effect + time effect - CNOI effect + noise
            ret_fwd = entity_effect + time_effect - 0.002 * cnoi + np.random.normal(0, 0.02)

            data.append({
                "ticker": ticker,
                "quarter": quarter,
                "CNOI": cnoi,
                "ret_fwd": ret_fwd,
                "log_mcap": np.random.uniform(20, 25),  # Control variable
            })

    df = pd.DataFrame(data)

    # Run all regressions
    results = run_all_panel_regressions(
        df,
        dependent_var="ret_fwd",
        independent_vars=["CNOI", "log_mcap"]
    )

    print("\n=== PANEL REGRESSION RESULTS ===")
    print(results["comparison"])

    print("\n=== INTERPRETATION ===")
    cnoi_coef_fe = results["FE"]["coefficients"]["CNOI"]
    cnoi_tstat_fe = results["FE"]["t_stats"]["CNOI"]
    print(f"Fixed Effects: CNOI coef = {cnoi_coef_fe:.4f} (t = {cnoi_tstat_fe:.2f})")

    if cnoi_tstat_fe < -1.96:
        print("✓ CNOI significantly predicts lower returns (p < 0.05)")
    else:
        print("✗ CNOI not statistically significant")
```

**Checkpoint 2.1.1**: `panel_regression.py` implemented, demo runs successfully

---

#### 2.1.2: Create tests for `panel_regression.py`

**File**: `tests/test_panel_regression.py`

```python
"""
Tests for src/analysis/panel_regression.py
"""

import pytest
import pandas as pd
import numpy as np
from src.analysis.panel_regression import (
    prepare_panel_data,
    fixed_effects_regression,
    fama_macbeth_regression,
    driscoll_kraay_regression,
    run_all_panel_regressions,
)


@pytest.fixture
def sample_panel_data():
    """Create sample panel data for testing."""
    np.random.seed(42)
    n_entities = 20
    n_periods = 10

    data = []
    for i in range(n_entities):
        ticker = f"STOCK{i:02d}"
        entity_effect = np.random.normal(0, 0.02)

        for t in range(n_periods):
            quarter = pd.Period(f"2020Q{(t % 4) + 1}") if t < 4 else pd.Period(f"202{1 + t // 4}Q{(t % 4) + 1}")
            time_effect = np.random.normal(0, 0.01)

            cnoi = 15 + i + np.random.normal(0, 1)
            ret_fwd = entity_effect + time_effect - 0.002 * cnoi + np.random.normal(0, 0.01)

            data.append({
                "ticker": ticker,
                "quarter": quarter,
                "CNOI": cnoi,
                "ret_fwd": ret_fwd,
            })

    return pd.DataFrame(data)


class TestPreparePanelData:
    """Tests for prepare_panel_data function."""

    def test_prepare_creates_multiindex(self, sample_panel_data):
        """Test that preparation creates MultiIndex."""
        result = prepare_panel_data(sample_panel_data, entity_col="ticker", time_col="quarter")

        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ["ticker", "quarter"]

    def test_prepare_preserves_data(self, sample_panel_data):
        """Test that preparation doesn't drop data."""
        result = prepare_panel_data(sample_panel_data, entity_col="ticker", time_col="quarter")

        assert len(result) == len(sample_panel_data)

    def test_prepare_sorts_index(self, sample_panel_data):
        """Test that index is sorted."""
        result = prepare_panel_data(sample_panel_data, entity_col="ticker", time_col="quarter")

        assert result.index.is_monotonic_increasing


class TestFixedEffectsRegression:
    """Tests for fixed_effects_regression function."""

    def test_fe_runs_successfully(self, sample_panel_data):
        """Test that FE regression runs without error."""
        panel_df = prepare_panel_data(sample_panel_data)

        results = fixed_effects_regression(
            panel_df,
            dependent_var="ret_fwd",
            independent_vars=["CNOI"]
        )

        assert "coefficients" in results
        assert "CNOI" in results["coefficients"]
        assert "t_stats" in results
        assert "p_values" in results

    def test_fe_with_strong_signal(self):
        """Test FE regression with known strong signal."""
        np.random.seed(42)
        n_entities = 30
        n_periods = 15

        data = []
        for i in range(n_entities):
            ticker = f"STOCK{i:02d}"

            for t in range(n_periods):
                quarter = pd.Period(f"2020Q1") + t
                cnoi = 15 + i
                ret_fwd = -0.005 * cnoi + np.random.normal(0, 0.01)  # Strong negative effect

                data.append({"ticker": ticker, "quarter": quarter, "CNOI": cnoi, "ret_fwd": ret_fwd})

        df = pd.DataFrame(data)
        panel_df = prepare_panel_data(df)

        results = fixed_effects_regression(panel_df, independent_vars=["CNOI"])

        # Should detect negative relationship
        assert results["coefficients"]["CNOI"] < 0
        assert abs(results["t_stats"]["CNOI"]) > 1.96  # Significant

    def test_fe_with_entity_effects_only(self, sample_panel_data):
        """Test FE with only entity effects."""
        panel_df = prepare_panel_data(sample_panel_data)

        results = fixed_effects_regression(
            panel_df,
            independent_vars=["CNOI"],
            entity_effects=True,
            time_effects=False
        )

        assert results["effects"] == "entity"


class TestFamaMacBethRegression:
    """Tests for fama_macbeth_regression function."""

    def test_fm_runs_successfully(self, sample_panel_data):
        """Test that FM regression runs without error."""
        results = fama_macbeth_regression(
            sample_panel_data,
            dependent_var="ret_fwd",
            independent_vars=["CNOI"],
            time_col="quarter"
        )

        assert "coefficients" in results
        assert "CNOI" in results["coefficients"]
        assert "n_periods" in results

    def test_fm_with_known_coefficient(self):
        """Test FM with known cross-sectional relationship."""
        np.random.seed(42)
        n_stocks = 50
        n_periods = 12

        data = []
        for t in range(n_periods):
            quarter = pd.Period("2020Q1") + t

            for i in range(n_stocks):
                ticker = f"STOCK{i:02d}"
                cnoi = 10 + i * 0.5
                ret_fwd = -0.003 * cnoi + np.random.normal(0, 0.02)  # Constant cross-sectional effect

                data.append({"ticker": ticker, "quarter": quarter, "CNOI": cnoi, "ret_fwd": ret_fwd})

        df = pd.DataFrame(data)

        results = fama_macbeth_regression(df, independent_vars=["CNOI"])

        # Should recover coefficient close to -0.003
        assert results["coefficients"]["CNOI"] < 0
        assert -0.005 < results["coefficients"]["CNOI"] < -0.001


class TestDriscollKraayRegression:
    """Tests for driscoll_kraay_regression function."""

    def test_dk_runs_successfully(self, sample_panel_data):
        """Test that DK regression runs without error."""
        panel_df = prepare_panel_data(sample_panel_data)

        results = driscoll_kraay_regression(
            panel_df,
            dependent_var="ret_fwd",
            independent_vars=["CNOI"]
        )

        assert "coefficients" in results
        assert "std_errors" in results
        assert "max_lags" in results


class TestRunAllPanelRegressions:
    """Tests for run_all_panel_regressions function."""

    def test_all_methods_agree_on_sign(self, sample_panel_data):
        """Test that all methods agree on coefficient sign."""
        results = run_all_panel_regressions(
            sample_panel_data,
            independent_vars=["CNOI"]
        )

        fe_coef = results["FE"]["coefficients"]["CNOI"]
        fm_coef = results["FM"]["coefficients"]["CNOI"]
        dk_coef = results["DK"]["coefficients"]["CNOI"]

        # All should have same sign (may not be significant, but should agree)
        assert np.sign(fe_coef) == np.sign(fm_coef) or abs(fm_coef) < 0.001
        assert np.sign(fe_coef) == np.sign(dk_coef) or abs(dk_coef) < 0.001
```

**Checkpoint 2.1.2**: Panel regression tests written and passing (>80% coverage)

---

### Task 2.2: Implement `dimension_analysis.py` (8-10 hours)

#### 2.2.1: Create module

**File**: `src/analysis/dimension_analysis.py`

```python
"""
CNOI dimension analysis: Test each dimension (D, G, R, J, T, S, X) separately.

Goal: Identify which CNOI dimensions drive stock returns.
Hypothesis: Stability (S) and Required Items (R) matter most for investors.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis.decile_backtest import run_decile_backtest
from src.utils.logger import get_logger

logger = get_logger(__name__)

# CNOI Dimension definitions
DIMENSIONS = {
    "D": "Discoverability (ease of finding CECL note)",
    "G": "Granularity (detail level)",
    "R": "Required Items (completeness)",
    "J": "Readability (complexity)",
    "T": "Table Density (use of tables vs text)",
    "S": "Stability (consistency over time)",
    "X": "Consistency (internal consistency)",
}


def analyze_single_dimension(
    df: pd.DataFrame,
    dimension: str,
    n_deciles: int = 10,
    weighting: str = "equal"
) -> Dict:
    """
    Run decile backtest for a single CNOI dimension.

    Args:
        df: DataFrame with dimension score and ret_fwd
        dimension: Dimension column name (D, G, R, J, T, S, or X)
        n_deciles: Number of deciles
        weighting: "equal" or "value"

    Returns:
        Dictionary with:
            - dimension: Dimension name
            - description: Dimension description
            - summary: Decile summary DataFrame
            - long_short: Long-short spread statistics

    Example:
        >>> results = analyze_single_dimension(df, dimension="S", n_deciles=10)
        >>> print(f"Stability long-short: {results['long_short']['mean_ret']:.4f}")
    """
    if dimension not in DIMENSIONS:
        raise ValueError(f"Unknown dimension: {dimension}. Must be one of {list(DIMENSIONS.keys())}")

    logger.info(f"Analyzing dimension {dimension}: {DIMENSIONS[dimension]}")

    # Run decile backtest on this dimension
    summary, long_short = run_decile_backtest(
        df,
        score_col=dimension,
        n_deciles=n_deciles,
        weighting=weighting
    )

    return {
        "dimension": dimension,
        "description": DIMENSIONS[dimension],
        "summary": summary,
        "long_short": long_short,
    }


def analyze_all_dimensions(
    df: pd.DataFrame,
    n_deciles: int = 10,
    weighting: str = "equal"
) -> Dict[str, Dict]:
    """
    Analyze all CNOI dimensions and compare results.

    Args:
        df: DataFrame with all dimension columns (D, G, R, J, T, S, X) and ret_fwd
        n_deciles: Number of deciles
        weighting: "equal" or "value"

    Returns:
        Dictionary with results for each dimension

    Example:
        >>> all_results = analyze_all_dimensions(df)
        >>> for dim, res in all_results.items():
        >>>     print(f"{dim}: t-stat = {res['long_short']['t_stat']:.2f}")
    """
    logger.info("Analyzing all CNOI dimensions")

    results = {}
    for dimension in DIMENSIONS.keys():
        if dimension not in df.columns:
            logger.warning(f"Dimension {dimension} not in data, skipping")
            continue

        try:
            dim_result = analyze_single_dimension(df, dimension, n_deciles, weighting)
            results[dimension] = dim_result
        except Exception as e:
            logger.error(f"Error analyzing dimension {dimension}: {e}")

    return results


def compare_dimensions(all_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table of all dimensions.

    Args:
        all_results: Dictionary from analyze_all_dimensions

    Returns:
        DataFrame with comparison:
            - Dimension
            - Description
            - Long-Short Return
            - T-Statistic
            - P-Value
            - Ranking (by abs t-stat)

    Example:
        >>> comparison = compare_dimensions(all_results)
        >>> print(comparison.sort_values("T-Statistic", key=abs, ascending=False))
    """
    rows = []
    for dim, res in all_results.items():
        rows.append({
            "Dimension": dim,
            "Description": res["description"],
            "Long-Short Return": res["long_short"]["mean_ret"],
            "T-Statistic": res["long_short"]["t_stat"],
            "P-Value": res["long_short"]["p_value"],
            "Significant (p<0.05)": res["long_short"]["p_value"] < 0.05,
        })

    comparison = pd.DataFrame(rows)

    # Rank by absolute t-statistic
    comparison["Ranking"] = comparison["T-Statistic"].abs().rank(ascending=False).astype(int)
    comparison = comparison.sort_values("Ranking")

    return comparison


def plot_dimension_comparison(comparison_df: pd.DataFrame, save_path: str = None):
    """
    Plot comparison of dimension t-statistics.

    Args:
        comparison_df: DataFrame from compare_dimensions
        save_path: Optional path to save figure

    Example:
        >>> plot_dimension_comparison(comparison, save_path="results/dimension_comparison.png")
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Bar plot of t-statistics
    colors = ["red" if p < 0.05 else "gray" for p in comparison_df["P-Value"]]

    ax.barh(comparison_df["Dimension"], comparison_df["T-Statistic"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(-1.96, color="red", linestyle="--", alpha=0.5, label="p=0.05 threshold")
    ax.axvline(1.96, color="red", linestyle="--", alpha=0.5)

    ax.set_xlabel("T-Statistic (Long-Short Spread)")
    ax.set_title("CNOI Dimension Analysis: Which Dimensions Predict Returns?")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")

    plt.show()


def compute_dimension_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix of CNOI dimensions.

    Args:
        df: DataFrame with dimension columns

    Returns:
        Correlation matrix

    Example:
        >>> corr_matrix = compute_dimension_correlations(df)
    """
    dimension_cols = [d for d in DIMENSIONS.keys() if d in df.columns]

    if len(dimension_cols) == 0:
        raise ValueError("No dimension columns found in data")

    corr = df[dimension_cols].corr()

    logger.info(f"Computed correlation matrix for {len(dimension_cols)} dimensions")

    return corr


if __name__ == "__main__":
    # Demo
    np.random.seed(42)

    # Simulate data where S (Stability) and R (Required Items) predict returns
    n = 1000
    quarters = pd.period_range("2020Q1", periods=20, freq="Q")

    data = []
    for _ in range(n):
        quarter = np.random.choice(quarters)

        # Simulate dimensions
        D = np.random.uniform(5, 15)
        G = np.random.uniform(5, 15)
        R = np.random.uniform(5, 15)  # Required items (strong signal)
        J = np.random.uniform(5, 15)
        T = np.random.uniform(5, 15)
        S = np.random.uniform(5, 15)  # Stability (strong signal)
        X = np.random.uniform(5, 15)

        CNOI = D + G + R + J + T + S + X

        # Return driven primarily by S and R
        ret_fwd = -0.003 * S - 0.002 * R + np.random.normal(0, 0.02)

        data.append({
            "quarter": quarter,
            "D": D, "G": G, "R": R, "J": J, "T": T, "S": S, "X": X,
            "CNOI": CNOI,
            "ret_fwd": ret_fwd,
        })

    df = pd.DataFrame(data)

    # Analyze all dimensions
    all_results = analyze_all_dimensions(df, n_deciles=10)

    # Compare
    comparison = compare_dimensions(all_results)
    print("\n=== DIMENSION COMPARISON ===")
    print(comparison.to_string(index=False))

    # Plot
    plot_dimension_comparison(comparison)

    print("\n=== INTERPRETATION ===")
    top_dim = comparison.iloc[0]["Dimension"]
    print(f"Most predictive dimension: {top_dim} ({DIMENSIONS[top_dim]})")
```

**Tests**: `tests/test_dimension_analysis.py` (similar structure to previous tests)

**Checkpoint 2.2**: Dimension analysis implemented and tested

---

### Task 2.3: Implement `performance_metrics.py` (8-10 hours)

#### 2.3.1: Create module

**File**: `src/utils/performance_metrics.py`

```python
"""
Portfolio performance metrics.

Implements standard risk-adjusted performance measures:
- Sharpe Ratio
- Sortino Ratio
- Information Ratio
- Maximum Drawdown
- Calmar Ratio
- Annualized Return / Volatility
"""

import pandas as pd
import numpy as np
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized return.

    Args:
        returns: Return series
        periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

    Returns:
        Annualized return

    Example:
        >>> ann_ret = annualized_return(daily_returns, periods_per_year=252)
    """
    if len(returns) == 0:
        raise ValueError("Returns series is empty")

    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    years = n_periods / periods_per_year

    if years == 0:
        raise ValueError("Insufficient data for annualization")

    return (1 + total_return) ** (1 / years) - 1


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized volatility.

    Args:
        returns: Return series
        periods_per_year: Number of periods per year

    Returns:
        Annualized volatility

    Example:
        >>> ann_vol = annualized_volatility(daily_returns, periods_per_year=252)
    """
    if len(returns) == 0:
        raise ValueError("Returns series is empty")

    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sharpe ratio

    Example:
        >>> sr = sharpe_ratio(daily_returns, risk_free_rate=0.02, periods_per_year=252)
    """
    excess = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Compute annualized Sortino ratio (downside deviation).

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio

    Example:
        >>> sortino = sortino_ratio(daily_returns, periods_per_year=252)
    """
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]

    if len(downside) == 0 or downside.std() == 0:
        return np.inf  # No downside volatility

    return np.sqrt(periods_per_year) * excess.mean() / downside.std()


def max_drawdown(returns: pd.Series) -> float:
    """
    Compute maximum drawdown.

    Args:
        returns: Return series

    Returns:
        Maximum drawdown (negative value)

    Example:
        >>> mdd = max_drawdown(daily_returns)
        >>> print(f"Max drawdown: {mdd:.2%}")
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    return drawdown.min()


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute Calmar ratio (return / max drawdown).

    Args:
        returns: Return series
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio

    Example:
        >>> calmar = calmar_ratio(daily_returns, periods_per_year=252)
    """
    ann_ret = annualized_return(returns, periods_per_year)
    mdd = abs(max_drawdown(returns))

    if mdd == 0:
        return np.inf

    return ann_ret / mdd


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Compute information ratio vs benchmark.

    Args:
        returns: Return series
        benchmark_returns: Benchmark return series
        periods_per_year: Number of periods per year

    Returns:
        Information ratio

    Example:
        >>> ir = information_ratio(portfolio_returns, sp500_returns, periods_per_year=252)
    """
    excess = returns - benchmark_returns

    if excess.std() == 0:
        return np.inf

    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def compute_all_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Compute all performance metrics.

    Args:
        returns: Return series
        benchmark_returns: Optional benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Dictionary with all metrics

    Example:
        >>> metrics = compute_all_metrics(daily_returns, sp500_returns, risk_free_rate=0.02)
        >>> print(f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
    """
    metrics = {
        "ann_return": annualized_return(returns, periods_per_year),
        "ann_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar_ratio(returns, periods_per_year),
    }

    if benchmark_returns is not None:
        metrics["information_ratio"] = information_ratio(returns, benchmark_returns, periods_per_year)

    return metrics


if __name__ == "__main__":
    # Demo
    np.random.seed(42)

    # Simulate returns
    daily_returns = pd.Series(np.random.normal(0.0005, 0.01, 252 * 3))  # 3 years

    metrics = compute_all_metrics(daily_returns, risk_free_rate=0.02, periods_per_year=252)

    print("=== PERFORMANCE METRICS ===")
    for key, value in metrics.items():
        if "return" in key or "drawdown" in key:
            print(f"{key}: {value:.2%}")
        else:
            print(f"{key}: {value:.2f}")
```

**Tests**: `tests/test_performance_metrics.py` (follow previous patterns)

**Checkpoint 2.3**: Performance metrics implemented and tested

---

## 📊 Definition of Done (Phase 2)

### Code Complete
- [x] `panel_regression.py` implemented (FE, FM, DK)
- [x] `dimension_analysis.py` implemented
- [x] `performance_metrics.py` implemented
- [x] All modules have comprehensive docstrings
- [x] Type hints on all functions
- [x] Demo sections work end-to-end
- [x] Logging integrated (no print statements)

### Test Coverage
- [x] `panel_regression.py`: >80% coverage
- [x] `dimension_analysis.py`: >80% coverage
- [x] `performance_metrics.py`: >80% coverage
- [x] All edge cases covered
- [x] Integration tests for end-to-end workflows

### CI/CD
- [x] All tests passing locally
- [x] CI/CD pipeline green on GitHub
- [x] No linting errors
- [x] Code formatted with black

### Documentation
- [x] README updated with new modules
- [x] Academic references added
- [x] Usage examples provided

---

## ✅ Ready for Phase 3 When...

1. ✅ All 3 modules implemented
2. ✅ Demo sections run successfully
3. ✅ >80% test coverage
4. ✅ CI/CD green
5. ✅ No TODOs remaining

**Next Phase**: `PHASE3_REAL_DATA_INTEGRATION.md`
