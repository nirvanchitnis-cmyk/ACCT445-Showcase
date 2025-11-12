import numpy as np
import pandas as pd
import pytest

from src.analysis.factor_models import standard_errors


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(42)
    n = 80
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    noise = rng.normal(scale=0.5, size=n)
    y = 0.2 + 0.5 * x1 - 0.3 * x2 + noise
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return pd.Series(y), X


@pytest.fixture(scope="module")
def short_regression_data():
    rng = np.random.default_rng(7)
    n = 20
    x = rng.normal(size=n)
    y = 0.1 + 0.4 * x + rng.normal(scale=0.2, size=n)
    X = pd.DataFrame({"x": x})
    return pd.Series(y), X


@pytest.fixture(scope="module")
def factor_data():
    rng = np.random.default_rng(123)
    dates = pd.date_range("2020-01-31", periods=120, freq="ME")
    factors = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0, 0.01, len(dates)),
            "SMB": rng.normal(0, 0.01, len(dates)),
            "HML": rng.normal(0, 0.01, len(dates)),
            "RMW": rng.normal(0, 0.01, len(dates)),
            "CMA": rng.normal(0, 0.01, len(dates)),
            "MOM": rng.normal(0, 0.01, len(dates)),
            "RF": np.full(len(dates), 0.0002),
        },
        index=dates,
    )
    portfolio = (
        0.001 + 0.4 * factors["Mkt-RF"] + 0.1 * factors["SMB"] + rng.normal(0, 0.01, len(dates))
    )
    return portfolio, factors


def test_fit_with_policy_auto_uses_hac(regression_data):
    y, X = regression_data
    result = standard_errors.fit_with_policy(y, X, se_type="auto")
    assert result.cov_type == "HAC"


def test_fit_with_policy_auto_with_cluster_ids_uses_twoway(regression_data):
    y, X = regression_data
    n = len(y)
    cluster_id = pd.Series(np.tile(np.arange(4), n // 4 + 1)[:n])
    time_id = pd.Series(np.tile(np.arange(5), n // 5 + 1)[:n])
    result = standard_errors.fit_with_policy(
        y,
        X,
        se_type="auto",
        cluster_id=cluster_id,
        time_id=time_id,
    )
    assert result.cov_type == "cluster"


def test_fit_with_policy_twoway_requires_identifiers(regression_data):
    y, X = regression_data
    with pytest.raises(ValueError):
        standard_errors.fit_with_policy(y, X, se_type="twoway")


def test_fit_with_policy_cluster_branch_requires_identifiers(regression_data):
    y, X = regression_data
    with pytest.raises(ValueError):
        standard_errors.fit_with_policy(y, X, se_type="cluster")


def test_fit_with_policy_cluster_branch_runs(regression_data):
    y, X = regression_data
    n = len(y)
    cluster_id = pd.Series(np.tile(np.arange(6), n // 6 + 1)[:n])
    result = standard_errors.fit_with_policy(y, X, se_type="cluster", cluster_id=cluster_id)
    assert result.cov_type == "cluster"


def test_fit_with_policy_robust_branch(short_regression_data):
    y, X = short_regression_data
    result = standard_errors.fit_with_policy(y, X, se_type="robust")
    assert result.cov_type == "HC3"


def test_fit_with_policy_auto_short_series_defaults_to_robust(short_regression_data):
    y, X = short_regression_data
    result = standard_errors.fit_with_policy(y, X, se_type="auto")
    assert result.cov_type == "HC3"


def test_fit_with_policy_accepts_numpy_inputs(regression_data):
    y, X_df = regression_data
    result = standard_errors.fit_with_policy(y, X_df.to_numpy(), se_type="hac")
    # Number of parameters should equal original columns + constant
    assert len(result.params) == X_df.shape[1] + 1


def test_fit_with_policy_auto_with_single_cluster_id(regression_data):
    y, X = regression_data
    n = len(y)
    cluster_id = pd.Series(np.tile(np.arange(3), n // 3 + 1)[:n])
    result = standard_errors.fit_with_policy(y, X, se_type="auto", cluster_id=cluster_id)
    assert result.cov_type == "cluster"


def test_fit_with_policy_ols_branch(regression_data):
    y, X = regression_data
    result = standard_errors.fit_with_policy(y, X, se_type="ols")
    assert result.cov_type == "nonrobust"


def test_fit_with_policy_invalid_se_type_raises(regression_data):
    y, X = regression_data
    with pytest.raises(ValueError):
        standard_errors.fit_with_policy(y, X, se_type="invalid")


def test_estimate_alpha_with_policy_ff5_model(factor_data):
    portfolio, factors = factor_data
    result = standard_errors.estimate_alpha_with_policy(
        portfolio, factors, model="FF5", se_type="hac"
    )
    assert result["n_obs"] == len(factors)
    assert set(result["factor_loadings"]) == {"Mkt-RF", "SMB", "HML", "RMW", "CMA"}


def test_estimate_alpha_with_policy_carhart_model(factor_data):
    portfolio, factors = factor_data
    result = standard_errors.estimate_alpha_with_policy(
        portfolio, factors, model="Carhart", se_type="hac"
    )
    assert set(result["factor_loadings"]) == {"Mkt-RF", "SMB", "HML", "MOM"}


def test_estimate_alpha_with_policy_ff3_without_rf_column(factor_data):
    portfolio, factors = factor_data
    factors_no_rf = factors.drop(columns=["RF"])
    result = standard_errors.estimate_alpha_with_policy(
        portfolio, factors_no_rf, model="FF3", se_type="hac"
    )
    assert set(result["factor_loadings"]) == {"Mkt-RF", "SMB", "HML"}


def test_estimate_alpha_with_policy_invalid_model(factor_data):
    portfolio, factors = factor_data
    with pytest.raises(ValueError):
        standard_errors.estimate_alpha_with_policy(portfolio, factors, model="XYZ")
