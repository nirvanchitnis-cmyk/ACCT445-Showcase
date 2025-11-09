Operations Guide
================

Daily Runner
------------

The automated update loop lives in ``src/runner/daily_backtest.py`` and can be
invoked locally or inside Docker.

.. code-block:: bash

   # Local run (executes immediately, then follows configured schedule)
   python src/runner/daily_backtest.py

   # One-shot execution for CI or smoke tests
   timeout 600s python src/runner/daily_backtest.py

Workflow:

1. Load ``config/sample_cnoi.csv`` (or production dataset referenced in config)
2. Enrich with tickers + fetch market data via ``fetch_bulk_data``
3. Run ``run_decile_backtest`` (lag + weighting from config)
4. Persist outputs to ``results/decile_summary_latest.csv`` and
   ``results/decile_long_short_latest.csv``
5. Trigger alerts when t-statistics fall below ``runner.anomaly_tstat_threshold``

Dashboard
---------

``src/dashboard/app.py`` exposes Streamlit views for overview stats, decile
results, risk metrics, the SVB event study, and data-quality monitoring.

.. code-block:: bash

   # Local preview
   streamlit run src/dashboard/app.py

   # Dockerized launch (matches docker-compose service)
   docker run --rm -p 8501:8501 \
     -v "$(pwd)/results:/app/results" \
     acct445-showcase:latest

The dashboard looks for ``*_latest`` CSVs first and gracefully falls back to
Phase 3 exports if those are missing.

Docker & Compose
----------------

* ``Dockerfile`` – production container with Python 3.11, requirements,
  Streamlit, Plotly, and Sphinx installed. Entry point starts the dashboard.
* ``docker-compose.yml`` – orchestrates two services:

  - ``acct445-app``: Dashboard (port 8501)
  - ``backtest-runner``: Invokes ``python src/runner/daily_backtest.py``

.. code-block:: bash

   docker compose up -d
   docker compose ps
   docker compose logs -f acct445-app
   docker compose down

All services mount ``./data``, ``./results``, ``./logs``, and ``./config``.

Logs & Monitoring
-----------------

* Structured logging uses ``src/utils/logger.get_logger``. After Task 5.7 the
  logger supports JSON output for production ingestion.
* Runner logs land in ``logs/runner.log`` (rotated per config).
* Dashboard and runner health checks are configured in docker-compose; Kubernetes
  or Docker swarm deployments can reuse the same endpoints.

Testing & Quality Gates
-----------------------

* ``pytest --cov=src --cov-report=term-missing --cov-fail-under=80``
* Pre-commit hook (``.pre-commit-config.yaml``) enforces Black, Ruff, whitespace,
  test coverage, and ``dvc status`` cleanliness.
* Sphinx docs must build successfully: ``make -C docs html``.

Data Versioning
---------------

DVC tracks all large CSV artifacts. Before running notebooks or dashboards:

.. code-block:: bash

   pip install -r requirements.txt
   dvc pull

After generating new artifacts:

.. code-block:: bash

   dvc add results/decile_summary_latest.csv
   git add results/decile_summary_latest.csv.dvc

Check ``results/checkpoints/`` for written progress reports (kept in git, not DVC).
