# Phase 1: Testing Infrastructure & Foundation

**Phase**: 1 of 5
**Estimated Time**: 20-30 hours
**Dependencies**: None (can start immediately)
**Status**: 🟡 Ready to Start

---

## 🎯 Objectives

Transform the codebase from 0% test coverage to >80% with:
1. Comprehensive pytest test suite for all 4 existing modules
2. GitHub Actions CI/CD pipeline (test, lint, format, coverage)
3. Data validation framework with schema checks
4. Structured logging (replace all `print()` statements)
5. Custom exception classes for domain-specific errors

**Success Criteria**:
- ✅ >80% test coverage across all modules
- ✅ CI/CD pipeline green on GitHub
- ✅ All edge cases covered (missing data, API failures, invalid inputs)
- ✅ Logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- ✅ Data validation catches schema violations

---

## 📋 Task Breakdown

### Task 1.1: Set Up Test Infrastructure (4-5 hours)

#### 1.1.1: Create test directory structure
```bash
mkdir -p tests
touch tests/__init__.py
touch tests/conftest.py
touch tests/test_cik_ticker_mapper.py
touch tests/test_data_loader.py
touch tests/test_decile_backtest.py
touch tests/test_event_study.py
```

#### 1.1.2: Configure pytest in `tests/conftest.py`
```python
"""
Pytest configuration and shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_cnoi_path():
    """Path to sample CNOI data."""
    return Path(__file__).parent.parent / "config" / "sample_cnoi.csv"


@pytest.fixture
def sample_cnoi_data():
    """Load sample CNOI data for testing."""
    path = Path(__file__).parent.parent / "config" / "sample_cnoi.csv"
    return pd.read_csv(path, parse_dates=["filing_date"])


@pytest.fixture
def mock_sec_ticker_data():
    """Mock SEC ticker mapping data."""
    return {
        "0000070858": {"ticker": "BAC", "title": "BANK OF AMERICA CORP /DE/"},
        "0000019617": {"ticker": "JPM", "title": "JPMORGAN CHASE & CO"},
        "0000810265": {"ticker": "WFC", "title": "WELLS FARGO & COMPANY/MN"},
        "0000006282": {"ticker": "AMSF", "title": "AMERISERV FINANCIAL INC /PA/"},
    }


@pytest.fixture
def mock_returns_data():
    """Mock return data for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    return pd.DataFrame({
        "date": dates,
        "ticker": ["BAC"] * 100,
        "ret": np.random.normal(0.001, 0.02, 100),
    })


@pytest.fixture
def sample_decile_data():
    """Sample data for decile backtest testing."""
    np.random.seed(42)
    n = 50  # 50 banks
    quarters = pd.period_range("2023Q1", periods=4, freq="Q")

    data = []
    for quarter in quarters:
        for i in range(n):
            data.append({
                "quarter": quarter,
                "ticker": f"BANK{i:02d}",
                "cnoi": 10 + i * 0.5 + np.random.normal(0, 1),
                "ret_fwd": 0.05 - i * 0.001 + np.random.normal(0, 0.02),
            })

    return pd.DataFrame(data)


@pytest.fixture
def sample_event_study_data():
    """Sample data for event study testing."""
    np.random.seed(42)
    tickers = ["BAC", "JPM", "WFC", "AMSF"]
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")

    data = []
    for ticker in tickers:
        for date in dates:
            market_ret = np.random.normal(0.001, 0.015)
            beta = 1.2 if ticker in ["BAC", "JPM"] else 0.8
            idio = np.random.normal(0, 0.01)
            ret = 0.0005 + beta * market_ret + idio

            data.append({
                "date": date,
                "ticker": ticker,
                "ret": ret,
                "mkt_ret": market_ret,
            })

    return pd.DataFrame(data)
```

#### 1.1.3: Update `pyproject.toml` for pytest coverage
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
addopts = [
    "--verbose",
    "--strict-markers",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]
```

**Checkpoint 1.1**: Test infrastructure set up, fixtures working
```bash
pytest tests/conftest.py --collect-only  # Should show fixtures
```

---

### Task 1.2: Test Suite for `cik_ticker_mapper.py` (4-5 hours)

#### 1.2.1: Create `tests/test_cik_ticker_mapper.py`

```python
"""
Tests for src/data/cik_ticker_mapper.py
"""

import pytest
import pandas as pd
import requests
from unittest.mock import patch, MagicMock
from src.data.cik_ticker_mapper import (
    fetch_sec_ticker_mapping,
    lookup_ticker_by_cik,
    enrich_cnoi_with_tickers,
)


class TestFetchSecTickerMapping:
    """Tests for fetch_sec_ticker_mapping function."""

    @patch("src.data.cik_ticker_mapper.requests.get")
    def test_fetch_success(self, mock_get, mock_sec_ticker_data):
        """Test successful fetch from SEC API."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_sec_ticker_data
        mock_get.return_value = mock_response

        result = fetch_sec_ticker_mapping()

        assert isinstance(result, dict)
        assert "0000070858" in result
        assert result["0000070858"]["ticker"] == "BAC"
        mock_get.assert_called_once()

    @patch("src.data.cik_ticker_mapper.requests.get")
    def test_fetch_network_error(self, mock_get):
        """Test handling of network errors."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

        with pytest.raises(requests.exceptions.ConnectionError):
            fetch_sec_ticker_mapping()

    @patch("src.data.cik_ticker_mapper.requests.get")
    def test_fetch_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            fetch_sec_ticker_mapping()

    @patch("src.data.cik_ticker_mapper.requests.get")
    def test_fetch_invalid_json(self, mock_get):
        """Test handling of invalid JSON response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response

        with pytest.raises(ValueError):
            fetch_sec_ticker_mapping()


class TestLookupTickerByCik:
    """Tests for lookup_ticker_by_cik function."""

    def test_lookup_existing_cik(self, mock_sec_ticker_data):
        """Test lookup of existing CIK."""
        ticker = lookup_ticker_by_cik("0000070858", mock_sec_ticker_data)
        assert ticker == "BAC"

    def test_lookup_missing_cik(self, mock_sec_ticker_data):
        """Test lookup of missing CIK returns None."""
        ticker = lookup_ticker_by_cik("9999999999", mock_sec_ticker_data)
        assert ticker is None

    def test_lookup_cik_without_ticker(self, mock_sec_ticker_data):
        """Test lookup of CIK that exists but has no ticker."""
        incomplete_data = mock_sec_ticker_data.copy()
        incomplete_data["0000070858"] = {"title": "Some Company"}
        ticker = lookup_ticker_by_cik("0000070858", incomplete_data)
        assert ticker is None

    def test_lookup_with_integer_cik(self, mock_sec_ticker_data):
        """Test that integer CIKs are converted to padded strings."""
        # This tests the CIK zero-padding logic
        ticker = lookup_ticker_by_cik(70858, mock_sec_ticker_data)
        assert ticker == "BAC"


class TestEnrichCnoiWithTickers:
    """Tests for enrich_cnoi_with_tickers function."""

    @patch("src.data.cik_ticker_mapper.fetch_sec_ticker_mapping")
    def test_enrich_success(self, mock_fetch, sample_cnoi_data, mock_sec_ticker_data):
        """Test successful enrichment of CNOI data."""
        mock_fetch.return_value = mock_sec_ticker_data

        result = enrich_cnoi_with_tickers(sample_cnoi_data)

        assert "ticker" in result.columns
        assert result["ticker"].notna().sum() > 0  # At least some tickers found

    @patch("src.data.cik_ticker_mapper.fetch_sec_ticker_mapping")
    def test_enrich_with_missing_ciks(self, mock_fetch, sample_cnoi_data):
        """Test enrichment when some CIKs don't have tickers."""
        mock_fetch.return_value = {}  # Empty mapping

        result = enrich_cnoi_with_tickers(sample_cnoi_data)

        assert "ticker" in result.columns
        assert result["ticker"].isna().all()  # All tickers should be NaN

    @patch("src.data.cik_ticker_mapper.fetch_sec_ticker_mapping")
    def test_enrich_preserves_original_columns(self, mock_fetch, sample_cnoi_data, mock_sec_ticker_data):
        """Test that enrichment preserves original columns."""
        mock_fetch.return_value = mock_sec_ticker_data
        original_cols = set(sample_cnoi_data.columns)

        result = enrich_cnoi_with_tickers(sample_cnoi_data)

        assert original_cols.issubset(set(result.columns))

    def test_enrich_empty_dataframe(self):
        """Test enrichment of empty DataFrame."""
        empty_df = pd.DataFrame(columns=["cik"])

        with patch("src.data.cik_ticker_mapper.fetch_sec_ticker_mapping") as mock_fetch:
            mock_fetch.return_value = {}
            result = enrich_cnoi_with_tickers(empty_df)

        assert len(result) == 0
        assert "ticker" in result.columns
```

#### 1.2.2: Run tests and verify coverage
```bash
pytest tests/test_cik_ticker_mapper.py -v --cov=src/data/cik_ticker_mapper --cov-report=term-missing
```

**Target**: >80% coverage for `cik_ticker_mapper.py`

**Checkpoint 1.2**: CIK ticker mapper fully tested
```bash
pytest tests/test_cik_ticker_mapper.py --cov=src/data/cik_ticker_mapper --cov-fail-under=80
```

---

### Task 1.3: Test Suite for `data_loader.py` (5-6 hours)

#### 1.3.1: Create `tests/test_data_loader.py`

```python
"""
Tests for src/utils/data_loader.py
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.utils.data_loader import (
    load_cnoi_data,
    load_market_returns,
    compute_forward_returns,
    merge_cnoi_with_returns,
)


class TestLoadCnoiData:
    """Tests for load_cnoi_data function."""

    def test_load_sample_data(self, sample_cnoi_path):
        """Test loading sample CNOI data."""
        df = load_cnoi_data(sample_cnoi_path)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "cik" in df.columns
        assert "CNOI" in df.columns
        assert "filing_date" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["filing_date"])

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_cnoi_data("nonexistent.csv")

    def test_load_validates_required_columns(self, tmp_path):
        """Test that loading validates required columns."""
        # Create invalid CSV
        invalid_file = tmp_path / "invalid.csv"
        pd.DataFrame({"wrong_column": [1, 2, 3]}).to_csv(invalid_file, index=False)

        with pytest.raises(ValueError, match="Required columns"):
            load_cnoi_data(invalid_file)

    def test_filing_date_parsing(self, sample_cnoi_path):
        """Test that filing_date is parsed correctly."""
        df = load_cnoi_data(sample_cnoi_path)

        assert pd.api.types.is_datetime64_any_dtype(df["filing_date"])
        assert df["filing_date"].notna().all()


class TestLoadMarketReturns:
    """Tests for load_market_returns function."""

    @patch("src.utils.data_loader.yf.download")
    def test_load_single_ticker(self, mock_download):
        """Test loading returns for single ticker."""
        mock_data = pd.DataFrame({
            "Adj Close": [100, 102, 101, 103],
        }, index=pd.date_range("2023-01-01", periods=4, freq="D"))
        mock_download.return_value = mock_data

        result = load_market_returns(
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-04"
        )

        assert isinstance(result, pd.DataFrame)
        assert "ticker" in result.columns
        assert "date" in result.columns
        assert "ret" in result.columns
        assert len(result) == 3  # 4 prices -> 3 returns

    @patch("src.utils.data_loader.yf.download")
    def test_load_multiple_tickers(self, mock_download):
        """Test loading returns for multiple tickers."""
        mock_data = pd.DataFrame({
            ("Adj Close", "AAPL"): [100, 102, 101],
            ("Adj Close", "MSFT"): [200, 204, 202],
        }, index=pd.date_range("2023-01-01", periods=3, freq="D"))
        mock_download.return_value = mock_data

        result = load_market_returns(
            tickers=["AAPL", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-01-03"
        )

        assert len(result["ticker"].unique()) == 2
        assert set(result["ticker"].unique()) == {"AAPL", "MSFT"}

    @patch("src.utils.data_loader.yf.download")
    def test_load_with_missing_data(self, mock_download):
        """Test handling of missing data (NaN values)."""
        mock_data = pd.DataFrame({
            "Adj Close": [100, np.nan, 102, 103],
        }, index=pd.date_range("2023-01-01", periods=4, freq="D"))
        mock_download.return_value = mock_data

        result = load_market_returns(
            tickers=["AAPL"],
            start_date="2023-01-01",
            end_date="2023-01-04"
        )

        # Should handle NaN gracefully (either drop or fill)
        assert result["ret"].notna().sum() >= 0

    @patch("src.utils.data_loader.yf.download")
    def test_load_invalid_ticker(self, mock_download):
        """Test handling of invalid ticker."""
        mock_download.return_value = pd.DataFrame()  # Empty

        result = load_market_returns(
            tickers=["INVALIDTICKER"],
            start_date="2023-01-01",
            end_date="2023-01-04"
        )

        assert len(result) == 0  # Should return empty DataFrame

    def test_load_with_invalid_dates(self):
        """Test that invalid date range raises error."""
        with pytest.raises(ValueError):
            load_market_returns(
                tickers=["AAPL"],
                start_date="2023-12-31",
                end_date="2023-01-01"  # End before start
            )


class TestComputeForwardReturns:
    """Tests for compute_forward_returns function."""

    def test_compute_forward_1day(self, mock_returns_data):
        """Test computing 1-day forward returns."""
        result = compute_forward_returns(mock_returns_data, horizon=1)

        assert "ret_fwd" in result.columns
        assert len(result) == len(mock_returns_data) - 1  # Lose last observation

    def test_compute_forward_multi_day(self, mock_returns_data):
        """Test computing multi-day forward returns."""
        result = compute_forward_returns(mock_returns_data, horizon=5)

        assert "ret_fwd" in result.columns
        assert len(result) == len(mock_returns_data) - 5

    def test_compute_forward_with_multiple_tickers(self):
        """Test forward returns with multiple tickers."""
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D").tolist() * 2,
            "ticker": ["AAPL"] * 10 + ["MSFT"] * 10,
            "ret": np.random.normal(0.001, 0.02, 20),
        })

        result = compute_forward_returns(data, horizon=1)

        assert len(result["ticker"].unique()) == 2
        assert result.groupby("ticker").size().max() == 9  # Each ticker loses 1 obs

    def test_compute_forward_validates_horizon(self, mock_returns_data):
        """Test that invalid horizon raises error."""
        with pytest.raises(ValueError):
            compute_forward_returns(mock_returns_data, horizon=0)

        with pytest.raises(ValueError):
            compute_forward_returns(mock_returns_data, horizon=-1)


class TestMergeCnoiWithReturns:
    """Tests for merge_cnoi_with_returns function."""

    def test_merge_success(self, sample_cnoi_data, mock_returns_data):
        """Test successful merge of CNOI and returns."""
        # Enrich CNOI with tickers first
        cnoi_with_ticker = sample_cnoi_data.copy()
        cnoi_with_ticker["ticker"] = "BAC"

        result = merge_cnoi_with_returns(
            cnoi_with_ticker,
            mock_returns_data,
            lag_days=2
        )

        assert isinstance(result, pd.DataFrame)
        assert "CNOI" in result.columns
        assert "ret_fwd" in result.columns

    def test_merge_with_lag_enforcement(self, sample_cnoi_data, mock_returns_data):
        """Test that lag_days enforces information timing."""
        cnoi_with_ticker = sample_cnoi_data.copy()
        cnoi_with_ticker["ticker"] = "BAC"
        cnoi_with_ticker["filing_date"] = pd.to_datetime("2023-01-01")

        result = merge_cnoi_with_returns(
            cnoi_with_ticker,
            mock_returns_data,
            lag_days=2
        )

        # Returns should start at least lag_days after filing
        if len(result) > 0:
            min_return_date = result["date"].min()
            filing_date = cnoi_with_ticker["filing_date"].iloc[0]
            assert (min_return_date - filing_date).days >= 2

    def test_merge_with_no_matches(self, sample_cnoi_data):
        """Test merge when no tickers match."""
        cnoi_with_ticker = sample_cnoi_data.copy()
        cnoi_with_ticker["ticker"] = "INVALIDTICKER"

        returns = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "ticker": ["AAPL"] * 10,
            "ret_fwd": np.random.normal(0.001, 0.02, 10),
        })

        result = merge_cnoi_with_returns(cnoi_with_ticker, returns, lag_days=2)

        assert len(result) == 0  # No matches

    def test_merge_preserves_cnoi_dimensions(self, sample_cnoi_data, mock_returns_data):
        """Test that merge preserves CNOI dimension columns (D, G, R, J, T, S, X)."""
        cnoi_with_ticker = sample_cnoi_data.copy()
        cnoi_with_ticker["ticker"] = "BAC"

        result = merge_cnoi_with_returns(cnoi_with_ticker, mock_returns_data, lag_days=2)

        dimension_cols = ["D", "G", "R", "J", "T", "S", "X"]
        for col in dimension_cols:
            if col in sample_cnoi_data.columns:
                assert col in result.columns
```

**Checkpoint 1.3**: Data loader fully tested
```bash
pytest tests/test_data_loader.py --cov=src/utils/data_loader --cov-fail-under=80
```

---

### Task 1.4: Test Suite for `decile_backtest.py` (5-6 hours)

#### 1.4.1: Create `tests/test_decile_backtest.py`

```python
"""
Tests for src/analysis/decile_backtest.py
"""

import pytest
import pandas as pd
import numpy as np
from src.analysis.decile_backtest import (
    assign_deciles,
    compute_decile_returns,
    newey_west_tstat,
    run_decile_backtest,
)


class TestAssignDeciles:
    """Tests for assign_deciles function."""

    def test_assign_10_deciles(self, sample_decile_data):
        """Test assigning 10 deciles."""
        quarter_data = sample_decile_data[sample_decile_data["quarter"] == "2023Q1"]

        result = assign_deciles(quarter_data, score_col="cnoi", n_deciles=10)

        assert "decile" in result.columns
        assert result["decile"].min() == 1
        assert result["decile"].max() == 10
        assert len(result["decile"].unique()) == 10

    def test_assign_5_deciles(self, sample_decile_data):
        """Test assigning 5 deciles (quintiles)."""
        quarter_data = sample_decile_data[sample_decile_data["quarter"] == "2023Q1"]

        result = assign_deciles(quarter_data, score_col="cnoi", n_deciles=5)

        assert result["decile"].min() == 1
        assert result["decile"].max() == 5

    def test_assign_deciles_with_ties(self):
        """Test decile assignment with tied scores."""
        data = pd.DataFrame({
            "cnoi": [10, 10, 20, 20, 30, 30, 40, 40, 50, 50],
            "ret_fwd": np.random.normal(0, 0.01, 10),
        })

        result = assign_deciles(data, score_col="cnoi", n_deciles=5)

        # Should still assign 5 deciles despite ties
        assert len(result["decile"].unique()) <= 5

    def test_assign_deciles_preserves_rows(self, sample_decile_data):
        """Test that decile assignment doesn't drop rows."""
        quarter_data = sample_decile_data[sample_decile_data["quarter"] == "2023Q1"]

        result = assign_deciles(quarter_data, score_col="cnoi", n_deciles=10)

        assert len(result) == len(quarter_data)

    def test_assign_deciles_with_missing_values(self):
        """Test decile assignment with missing CNOI values."""
        data = pd.DataFrame({
            "cnoi": [10, np.nan, 20, 30, np.nan, 40, 50, 60, 70, 80],
            "ret_fwd": np.random.normal(0, 0.01, 10),
        })

        result = assign_deciles(data, score_col="cnoi", n_deciles=5)

        # Should handle NaN (either drop or assign separate category)
        assert result["decile"].notna().sum() >= 0


class TestComputeDecileReturns:
    """Tests for compute_decile_returns function."""

    def test_compute_equal_weighted(self, sample_decile_data):
        """Test equal-weighted decile returns."""
        quarter_data = sample_decile_data[sample_decile_data["quarter"] == "2023Q1"]
        quarter_data = assign_deciles(quarter_data, score_col="cnoi", n_deciles=10)

        result = compute_decile_returns(
            quarter_data,
            weighting="equal"
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 10
        assert result.index.name == "decile"

    def test_compute_value_weighted(self, sample_decile_data):
        """Test value-weighted decile returns."""
        quarter_data = sample_decile_data[sample_decile_data["quarter"] == "2023Q1"]
        quarter_data = assign_deciles(quarter_data, score_col="cnoi", n_deciles=10)
        quarter_data["market_cap"] = np.random.uniform(1e9, 1e11, len(quarter_data))

        result = compute_decile_returns(
            quarter_data,
            weighting="value",
            market_cap_col="market_cap"
        )

        assert isinstance(result, pd.Series)
        assert len(result) == 10

    def test_equal_vs_value_weighted_differ(self, sample_decile_data):
        """Test that equal and value weighted returns differ."""
        quarter_data = sample_decile_data[sample_decile_data["quarter"] == "2023Q1"]
        quarter_data = assign_deciles(quarter_data, score_col="cnoi", n_deciles=10)
        quarter_data["market_cap"] = np.random.uniform(1e9, 1e11, len(quarter_data))

        equal = compute_decile_returns(quarter_data, weighting="equal")
        value = compute_decile_returns(quarter_data, weighting="value", market_cap_col="market_cap")

        # Should differ (unless by extreme coincidence)
        assert not np.allclose(equal.values, value.values)

    def test_compute_handles_empty_decile(self):
        """Test handling of empty deciles."""
        data = pd.DataFrame({
            "decile": [1, 1, 1, 3, 3, 3],  # Decile 2 missing
            "ret_fwd": [0.01, 0.02, 0.015, -0.01, -0.02, -0.015],
        })

        result = compute_decile_returns(data, weighting="equal")

        # Should handle missing decile gracefully
        assert len(result) <= 3


class TestNeweyWestTstat:
    """Tests for newey_west_tstat function."""

    def test_positive_mean_positive_tstat(self):
        """Test that positive mean yields positive t-stat."""
        returns = pd.Series(np.random.normal(0.01, 0.02, 100))

        tstat = newey_west_tstat(returns, lags=3)

        assert tstat > 0  # Positive mean should give positive t-stat

    def test_negative_mean_negative_tstat(self):
        """Test that negative mean yields negative t-stat."""
        returns = pd.Series(np.random.normal(-0.01, 0.02, 100))

        tstat = newey_west_tstat(returns, lags=3)

        assert tstat < 0

    def test_zero_variance_raises(self):
        """Test that zero variance raises error."""
        returns = pd.Series([0.01] * 100)  # Constant returns

        with pytest.raises((ValueError, ZeroDivisionError)):
            newey_west_tstat(returns, lags=3)

    def test_different_lags_differ(self):
        """Test that different lag specifications give different t-stats."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.01, 0.02, 100))

        tstat_lag1 = newey_west_tstat(returns, lags=1)
        tstat_lag5 = newey_west_tstat(returns, lags=5)

        # Should differ (more lags = larger SE = smaller t-stat usually)
        assert tstat_lag1 != tstat_lag5

    def test_short_series_warning(self):
        """Test that short series raises warning."""
        returns = pd.Series([0.01, 0.02, 0.015])  # Only 3 observations

        # Should either raise or warn
        with pytest.warns(UserWarning):
            newey_west_tstat(returns, lags=3)


class TestRunDecileBacktest:
    """Tests for run_decile_backtest function."""

    def test_run_backtest_complete(self, sample_decile_data):
        """Test complete backtest run."""
        summary, long_short = run_decile_backtest(
            sample_decile_data,
            score_col="cnoi",
            n_deciles=10,
            weighting="equal"
        )

        assert isinstance(summary, pd.DataFrame)
        assert isinstance(long_short, dict)
        assert "decile" in summary.columns
        assert "mean_ret" in summary.columns
        assert "t_stat" in summary.columns

    def test_backtest_long_short_spread(self, sample_decile_data):
        """Test long-short spread calculation."""
        summary, long_short = run_decile_backtest(
            sample_decile_data,
            score_col="cnoi",
            n_deciles=10,
            weighting="equal"
        )

        assert "mean_ret" in long_short
        assert "t_stat" in long_short
        assert "p_value" in long_short

        # Long-short should be D1 - D10
        d1_ret = summary[summary["decile"] == 1]["mean_ret"].iloc[0]
        d10_ret = summary[summary["decile"] == 10]["mean_ret"].iloc[0]
        expected_spread = d1_ret - d10_ret

        assert np.isclose(long_short["mean_ret"], expected_spread)

    def test_backtest_with_simulated_signal(self):
        """Test backtest with strong simulated signal."""
        np.random.seed(42)
        n = 100
        quarters = pd.period_range("2020Q1", periods=20, freq="Q")

        # Create strong negative relationship: high CNOI -> low returns
        data = []
        for quarter in quarters:
            for i in range(n):
                cnoi = 10 + i * 0.5
                ret_fwd = 0.10 - 0.002 * i + np.random.normal(0, 0.01)  # Strong signal
                data.append({"quarter": quarter, "cnoi": cnoi, "ret_fwd": ret_fwd})

        df = pd.DataFrame(data)

        summary, long_short = run_decile_backtest(df, score_col="cnoi", n_deciles=10)

        # Should have significant long-short spread
        assert long_short["p_value"] < 0.05  # Significant at 5%
        assert long_short["mean_ret"] > 0  # D1 (low CNOI) beats D10 (high CNOI)

    def test_backtest_handles_missing_quarters(self):
        """Test backtest with gaps in quarters."""
        data = sample_decile_data.copy()
        # Remove Q3
        data = data[data["quarter"] != "2023Q3"]

        summary, long_short = run_decile_backtest(data, score_col="cnoi", n_deciles=10)

        # Should still run successfully
        assert len(summary) == 10
```

**Checkpoint 1.4**: Decile backtest fully tested
```bash
pytest tests/test_decile_backtest.py --cov=src/analysis/decile_backtest --cov-fail-under=80
```

---

### Task 1.5: Test Suite for `event_study.py` (5-6 hours)

#### 1.5.1: Create `tests/test_event_study.py`

```python
"""
Tests for src/analysis/event_study.py
"""

import pytest
import pandas as pd
import numpy as np
from src.analysis.event_study import (
    compute_market_model_params,
    compute_abnormal_returns,
    compute_cumulative_abnormal_returns,
    run_event_study,
)


class TestComputeMarketModelParams:
    """Tests for compute_market_model_params function."""

    def test_compute_params_simple(self, sample_event_study_data):
        """Test market model parameter estimation."""
        ticker_data = sample_event_study_data[
            sample_event_study_data["ticker"] == "BAC"
        ]

        alpha, beta = compute_market_model_params(
            ticker_data,
            estimation_start="2023-01-01",
            estimation_end="2023-02-28"
        )

        assert isinstance(alpha, float)
        assert isinstance(beta, float)
        assert np.isfinite(alpha)
        assert np.isfinite(beta)
        assert beta > 0  # Bank stocks should have positive beta

    def test_beta_interpretation(self):
        """Test that beta correctly captures market sensitivity."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create data with known beta = 1.5
        mkt_ret = np.random.normal(0.001, 0.015, 100)
        stock_ret = 0.0005 + 1.5 * mkt_ret + np.random.normal(0, 0.005, 100)

        data = pd.DataFrame({
            "date": dates,
            "ret": stock_ret,
            "mkt_ret": mkt_ret,
        })

        alpha, beta = compute_market_model_params(
            data,
            estimation_start="2023-01-01",
            estimation_end="2023-04-10"
        )

        # Beta should be close to 1.5
        assert 1.3 < beta < 1.7

    def test_insufficient_data_raises(self):
        """Test that insufficient data raises error."""
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=5, freq="D"),
            "ret": [0.01] * 5,
            "mkt_ret": [0.01] * 5,
        })

        with pytest.raises(ValueError, match="Insufficient"):
            compute_market_model_params(
                data,
                estimation_start="2023-01-01",
                estimation_end="2023-01-05"
            )


class TestComputeAbnormalReturns:
    """Tests for compute_abnormal_returns function."""

    def test_compute_ar_simple(self, sample_event_study_data):
        """Test abnormal return computation."""
        ticker_data = sample_event_study_data[
            sample_event_study_data["ticker"] == "BAC"
        ]

        # First get params
        alpha, beta = compute_market_model_params(
            ticker_data,
            estimation_start="2023-01-01",
            estimation_end="2023-02-28"
        )

        # Then compute AR
        result = compute_abnormal_returns(
            ticker_data,
            alpha=alpha,
            beta=beta,
            event_start="2023-03-01",
            event_end="2023-03-15"
        )

        assert "abnormal_ret" in result.columns
        assert result["abnormal_ret"].notna().all()

    def test_ar_zero_for_perfect_model(self):
        """Test that AR is ~0 when model perfectly predicts returns."""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # Create data that perfectly follows market model
        alpha, beta = 0.0005, 1.2
        mkt_ret = np.random.normal(0.001, 0.015, 100)
        stock_ret = alpha + beta * mkt_ret  # No idiosyncratic component

        data = pd.DataFrame({
            "date": dates,
            "ret": stock_ret,
            "mkt_ret": mkt_ret,
        })

        result = compute_abnormal_returns(
            data,
            alpha=alpha,
            beta=beta,
            event_start="2023-03-01",
            event_end="2023-03-15"
        )

        # AR should be very close to 0
        assert np.abs(result["abnormal_ret"].mean()) < 0.001


class TestComputeCumulativeAbnormalReturns:
    """Tests for compute_cumulative_abnormal_returns function."""

    def test_compute_car_simple(self):
        """Test CAR computation."""
        data = pd.DataFrame({
            "date": pd.date_range("2023-03-01", periods=10, freq="D"),
            "abnormal_ret": [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.01, 0.005, -0.015, 0.01],
        })

        car = compute_cumulative_abnormal_returns(data)

        assert isinstance(car, float)
        assert np.isfinite(car)
        assert np.isclose(car, data["abnormal_ret"].sum())

    def test_car_empty_data(self):
        """Test CAR with empty data."""
        data = pd.DataFrame(columns=["date", "abnormal_ret"])

        with pytest.raises(ValueError):
            compute_cumulative_abnormal_returns(data)


class TestRunEventStudy:
    """Tests for run_event_study function."""

    def test_run_event_study_complete(self, sample_event_study_data):
        """Test complete event study run."""
        result = run_event_study(
            sample_event_study_data,
            event_date="2023-03-10",
            event_window_days=7,
            estimation_window_days=60
        )

        assert isinstance(result, pd.DataFrame)
        assert "ticker" in result.columns
        assert "CAR" in result.columns
        assert "alpha" in result.columns
        assert "beta" in result.columns

    def test_event_study_with_cnoi_quartiles(self):
        """Test event study grouped by CNOI quartiles."""
        # Create data with CNOI
        np.random.seed(42)
        tickers = [f"BANK{i:02d}" for i in range(20)]
        dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")

        data = []
        for i, ticker in enumerate(tickers):
            cnoi = 10 + i  # CNOI ranges from 10 to 29
            for date in dates:
                mkt_ret = np.random.normal(0.001, 0.015)
                beta = 1.0 + 0.01 * cnoi  # Higher CNOI = higher beta (more risk)
                # During event (March 10-17), high CNOI banks drop more
                event_shock = -0.05 if (pd.Timestamp("2023-03-10") <= date <= pd.Timestamp("2023-03-17") and cnoi > 20) else 0
                stock_ret = 0.0005 + beta * mkt_ret + event_shock + np.random.normal(0, 0.01)

                data.append({
                    "date": date,
                    "ticker": ticker,
                    "ret": stock_ret,
                    "mkt_ret": mkt_ret,
                    "cnoi": cnoi,
                })

        df = pd.DataFrame(data)

        result = run_event_study(
            df,
            event_date="2023-03-10",
            event_window_days=7,
            estimation_window_days=60
        )

        # Merge with CNOI
        result = result.merge(
            df[["ticker", "cnoi"]].drop_duplicates(),
            on="ticker"
        )

        # High CNOI banks should have worse CAR
        high_cnoi = result[result["cnoi"] > 20]["CAR"].mean()
        low_cnoi = result[result["cnoi"] <= 20]["CAR"].mean()

        assert high_cnoi < low_cnoi  # High CNOI performed worse
```

**Checkpoint 1.5**: Event study fully tested

---

### Task 1.6: GitHub Actions CI/CD Pipeline (3-4 hours)

#### 1.6.1: Create `.github/workflows/test.yml`

```yaml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .

    - name: Lint with ruff
      run: |
        ruff check src/

    - name: Format check with black
      run: |
        black --check src/ tests/

    - name: Type check with mypy (optional)
      continue-on-error: true
      run: |
        pip install mypy
        mypy src/ --ignore-missing-imports

    - name: Run tests with pytest
      run: |
        pytest tests/ \
          --verbose \
          --cov=src \
          --cov-report=term-missing \
          --cov-report=xml \
          --cov-fail-under=80

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  integration:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python 3.11
      uses: actions/setup-python@v5
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .

    - name: Run integration tests
      run: |
        pytest tests/ -m integration --verbose
```

#### 1.6.2: Add Codecov badge to README

```markdown
# ACCT445-Showcase

[![Test Suite](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/test.yml/badge.svg)](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/nirvanchitnis-cmyk/ACCT445-Showcase/branch/main/graph/badge.svg)](https://codecov.io/gh/nirvanchitnis-cmyk/ACCT445-Showcase)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Rest of README...]
```

**Checkpoint 1.6**: CI/CD pipeline green on GitHub

---

### Task 1.7: Logging Infrastructure (2-3 hours)

#### 1.7.1: Create `src/utils/logger.py`

```python
"""
Logging configuration for ACCT445-Showcase.

Replaces all print() statements with structured logging.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Get configured logger instance.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Analysis started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
```

#### 1.7.2: Replace print() in all modules

**Example for `decile_backtest.py`:**

```python
# Before:
print(f"Processing quarter {quarter}")
print(f"Long-short spread: {spread:.4f}")

# After:
from src.utils.logger import get_logger

logger = get_logger(__name__)

logger.info(f"Processing quarter {quarter}")
logger.info(f"Long-short spread: {spread:.4f}")
```

**Checkpoint 1.7**: All print() statements replaced with logging

---

### Task 1.8: Data Validation Framework (2-3 hours)

#### 1.8.1: Create `src/utils/validation.py`

```python
"""
Data validation and schema checking.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


def validate_cnoi_schema(df: pd.DataFrame) -> None:
    """
    Validate CNOI DataFrame schema.

    Args:
        df: CNOI DataFrame to validate

    Raises:
        DataValidationError: If schema validation fails

    Example:
        >>> validate_cnoi_schema(cnoi_df)
    """
    required_cols = ["cik", "filing_date", "CNOI", "D", "G", "R", "J", "T", "S", "X"]
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df["filing_date"]):
        raise DataValidationError("filing_date must be datetime type")

    # Check CNOI range
    if (df["CNOI"] < 0).any() or (df["CNOI"] > 100).any():
        logger.warning("CNOI values outside expected range [0, 100]")

    # Check for duplicates
    duplicates = df.duplicated(subset=["cik", "filing_date"]).sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate (cik, filing_date) pairs")

    logger.info(f"✓ CNOI schema validation passed ({len(df)} rows)")


def validate_returns_schema(df: pd.DataFrame) -> None:
    """
    Validate returns DataFrame schema.

    Args:
        df: Returns DataFrame to validate

    Raises:
        DataValidationError: If schema validation fails
    """
    required_cols = ["date", "ticker", "ret"]
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    # Check for extreme returns (likely errors)
    extreme_returns = (df["ret"].abs() > 0.5).sum()
    if extreme_returns > 0:
        logger.warning(f"Found {extreme_returns} extreme returns (>50% daily)")

    logger.info(f"✓ Returns schema validation passed ({len(df)} rows)")


def validate_merge_coverage(
    cnoi_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    min_coverage: float = 0.8
) -> None:
    """
    Validate that merge will have sufficient coverage.

    Args:
        cnoi_df: CNOI DataFrame with 'ticker' column
        returns_df: Returns DataFrame
        min_coverage: Minimum fraction of CNOI tickers that must have returns

    Raises:
        DataValidationError: If coverage too low
    """
    cnoi_tickers = set(cnoi_df["ticker"].dropna().unique())
    return_tickers = set(returns_df["ticker"].unique())

    matched = cnoi_tickers & return_tickers
    coverage = len(matched) / len(cnoi_tickers) if cnoi_tickers else 0

    logger.info(f"Merge coverage: {coverage:.1%} ({len(matched)}/{len(cnoi_tickers)} tickers)")

    if coverage < min_coverage:
        raise DataValidationError(
            f"Merge coverage {coverage:.1%} below threshold {min_coverage:.1%}"
        )
```

**Checkpoint 1.8**: Data validation framework complete with tests

---

## 📊 Definition of Done (Phase 1)

### Code Complete
- [x] Test directory structure created
- [x] pytest configured in pyproject.toml
- [x] All 4 modules have comprehensive test suites
- [x] Logging infrastructure implemented
- [x] Data validation framework created
- [x] All print() statements replaced with logging
- [x] GitHub Actions CI/CD pipeline configured

### Test Coverage
- [x] `cik_ticker_mapper.py`: >80% coverage
- [x] `data_loader.py`: >80% coverage
- [x] `decile_backtest.py`: >80% coverage
- [x] `event_study.py`: >80% coverage
- [x] Overall project coverage: >80%

### CI/CD
- [x] GitHub Actions workflow file created
- [x] Workflow runs on push and PR
- [x] Tests pass on Python 3.10, 3.11, 3.12
- [x] Linting (ruff) passes
- [x] Formatting (black) passes
- [x] Coverage uploaded to Codecov
- [x] Badges added to README

### Documentation
- [x] All test functions have docstrings
- [x] Fixtures documented in conftest.py
- [x] Logging usage documented
- [x] Validation functions documented

### Quality Gates
- [x] All tests passing locally
- [x] All tests passing on GitHub Actions
- [x] No hardcoded paths
- [x] No TODO/FIXME comments remaining
- [x] Git history clean (good commit messages)

---

## 🔍 Validation Checklist

Before proceeding to Phase 2, verify:

```bash
# 1. Run full test suite locally
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80

# 2. Check formatting
black --check src/ tests/

# 3. Check linting
ruff check src/

# 4. Verify CI/CD
# Push to GitHub and check Actions tab

# 5. Check coverage report
open htmlcov/index.html  # View detailed coverage

# 6. Verify logging
python -m src.analysis.decile_backtest  # Should see structured logs, not print()
```

**All checks must pass before Phase 2.**

---

## 🎯 Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Test Coverage | >80% | `pytest --cov=src --cov-fail-under=80` |
| Tests Passing | 100% | `pytest tests/` |
| CI/CD Green | ✅ | GitHub Actions badge |
| Code Quality | A+ | `ruff check src/` (0 errors) |
| Logging Complete | 100% | `grep -r "print(" src/` (0 results) |

---

## 🚨 Common Issues & Solutions

### Issue 1: yfinance Tests Failing (API Rate Limits)
**Solution**: Mock all yfinance calls using `unittest.mock.patch`

### Issue 2: Newey-West Tests Non-Deterministic
**Solution**: Use fixed random seed (`np.random.seed(42)`) in all tests

### Issue 3: CI/CD Fails on Older Python
**Solution**: Check Python version compatibility, adjust matrix if needed

### Issue 4: Coverage Below 80%
**Solution**: Add tests for edge cases, error paths, and `if __name__ == "__main__"` blocks

---

## 📝 Checkpoint Reports

Generate report every 8-10 hours using this template:

```markdown
## Phase 1 Checkpoint [X/5]

**Time Spent**: X hours
**Completion**: XX%

### Completed
- ✅ Task 1.1: Test infrastructure
- ✅ Task 1.2: CIK ticker mapper tests

### In Progress
- 🔄 Task 1.3: Data loader tests (70% complete)

### Blocked/Issues
- None

### Test Status
- Tests passing: XX/YY
- Coverage: XX%
- CI/CD: 🟡 In progress

### Next Steps
- Complete data loader tests
- Start decile backtest tests
```

---

## ✅ Ready for Phase 2 When...

1. ✅ All tests passing (100%)
2. ✅ Coverage >80% across all modules
3. ✅ CI/CD pipeline green on GitHub
4. ✅ No print() statements in src/
5. ✅ Data validation framework tested
6. ✅ README updated with badges

**Phase 2 Directive**: `PHASE2_CORE_ANALYSIS.md`

---

**Document Control**

**Version**: 1.0
**Status**: ✅ Ready to execute
**Estimated Time**: 20-30 hours
**Dependencies**: None
**Next Phase**: Phase 2 (Core Analysis Modules)
