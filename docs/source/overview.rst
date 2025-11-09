Overview
========

Purpose
-------

The ACCT445 Showcase system monitors disclosure opacity for US banks and turns
that signal into tradeable insights. It ingests company filing metadata,
calculates composite and dimension-specific CNOI scores, joins them with market
returns, and publishes the results through:

* Automated daily backtests (Task 5.2 runner)
* Streamlit dashboard (Task 5.3)
* Research notebooks from Phase 3 (exploration, deciles, event study, panel regression)

Architecture
------------

.. list-table:: Core components
   :header-rows: 1

   * - Layer
     - Modules / Files
     - Responsibilities
   * - Data Acquisition
     - ``src/data/sec_api_client.py``, ``src/utils/market_data.py``
     - Pull SEC metadata + yfinance prices with caching + rate limiting
   * - Feature Engineering
     - ``src/data/cik_ticker_mapper.py``, ``src/utils/data_loader.py``
     - Map CIK→ticker, align filings with market windows, compute forward returns
   * - Analytics
     - ``src/analysis/decile_backtest.py``, ``src/analysis/dimension_analysis.py``,
       ``src/analysis/event_study.py``, ``src/analysis/panel_regression.py``
     - Run decile, dimension, event, and panel studies
   * - Robustness & Costs
     - ``src/analysis/robustness.py``, ``src/utils/transaction_costs.py``
     - Bootstrap/permutation checks, transaction-cost adjustments
   * - Operations
     - ``src/runner/daily_backtest.py``, ``src/dashboard/app.py``
     - Scheduled updates, dashboard monitoring, alerting

Data Products
-------------

All outputs are written under ``results/`` and tracked with DVC:

* ``cnoi_with_tickers.csv`` – latest enriched filings
* ``decile_summary_*`` – decile performance tables
* ``decile_long_short_*`` – time-series spreads for dashboard metrics
* ``event_study_*.csv`` – SVB crisis abnormal returns
* ``panel_regression_results.csv`` – FE/FM/DK coefficient tables

Use ``dvc pull`` to download shared artifacts before running notebooks or the dashboard.

Dependencies
------------

* Python 3.11
* System: git, curl (Dockerfile installs)
* Python libs: see ``requirements.txt`` (notably pandas, numpy, statsmodels, linearmodels, streamlit, plotly, sphinx)
* External: Internet access for SEC/yfinance (production runner uses cached data whenever possible)

Configuration
-------------

All tunable parameters live in ``config/config.toml`` and are accessible via
``src/utils/config.get_config_value``. Key sections:

* ``[data]`` – file paths for CNOI data, cache directories, results dir
* ``[backtest]`` – number of deciles, weighting scheme, lag horizons
* ``[transaction_costs]`` – spread/impact coefficients
* ``[runner]`` – schedule time, anomaly thresholds
* ``[alerts]`` – email toggles/placeholders for future SMTP integration

Update the config file instead of hardcoding constants to keep modules flexible.
