# ACCT 445 · Bank Disclosure Opacity and Market Performance

[![Tests](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/test.yml/badge.svg)](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/test.yml)
[![Accessibility](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/accessibility.yml/badge.svg)](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/accessibility.yml)
[![Lighthouse](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/lighthouse.yml/badge.svg)](https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase/actions/workflows/lighthouse.yml)
[![codecov](https://codecov.io/gh/nirvanchitnis-cmyk/ACCT445-Showcase/branch/main/graph/badge.svg)](https://codecov.io/gh/nirvanchitnis-cmyk/ACCT445-Showcase)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

Live site: https://nirvanchitnis-cmyk.github.io/ACCT445-Showcase/

## Abstract

This repository studies whether disclosure opacity in banks’ CECL notes predicts subsequent stock returns and risk. We develop the CECL Note Opacity Index (CNOI), compute it for SEC filers, and evaluate performance with decile backtests, event‑study methods, and factor‑adjusted alphas. The codebase is fully reproducible (tests, coverage, data checks) and includes optional Docker deployment with a monitoring dashboard.

## Key Results (summary)

- Opaque banks underperform more transparent banks by ~200 bps per quarter on average.
- During the March 2023 SVB episode, the most opaque quartile experienced materially worse CAR than the most transparent quartile.
- The opacity premium remains statistically significant after Fama–French/Carhart adjustments.

## Data and Methods

- Sample: 50 banks, 509 filings (10‑K/10‑Q), 2023–2025.
- CNOI dimensions (weights in parentheses): Discoverability (20), Granularity (20), Required Items (20), Readability (10), Table Density (10), Stability (10), Consistency (10).
- Methods overview:
  - Decile backtest with quarterly rebalances and Newey–West standard errors.
  - Event study (SVB window) with robust tests (BMP, Corrado, sign).
  - Factor models (FF3/FF5/Carhart) and alpha decomposition.
  - Panel regressions with Driscoll–Kraay SEs; DiD variants with two‑way clustering.

For full details, see `docs/METHODOLOGY.md`.

## Repository Layout

```
src/
  analysis/        # Backtests, event studies, factor models, panel
  data/            # SEC API, market data helpers
  dashboard/       # Streamlit monitoring app
  runner/          # Daily pipeline + alerts
  utils/           # Logging, caching, validation, factor utilities
tests/             # Unit/integration tests (coverage ≥ 80%)
notebooks/         # Exploration and reporting notebooks
results/           # CSV outputs (DVC‑tracked where appropriate)
config/            # Sample data + runtime configuration
```

## Getting Started

Prerequisites: Python 3.11+, git, optional Docker.

Install and test locally:

```bash
git clone https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase.git
cd ACCT445-Showcase
pip install -r requirements.txt
pre-commit install
pytest -q
```

Reproduce the pipeline (DVC + tests):

```bash
make install        # deps + pre-commit + dvc pull
make test           # full test suite with coverage gate (≥ 80%)
make reproduce      # dvc repro (rebuild data products)
```

## Running with Docker

```bash
docker-compose up -d
# Dashboard: http://localhost:8501
```

Services:
- Dashboard (Streamlit, port 8501)
- Daily runner (scheduled updates, JSON logging with rotation)

See `DEPLOYMENT.md` for production guidance.

## Reproducibility and Quality

- Tests: Pytest with coverage threshold (80%); CI enforced via GitHub Actions.
- Linting/formatting: Ruff and Black (pre‑commit hooks).
- Data versioning: DVC for caches and generated CSVs.
- Integrity: Factor checksums and provenance utilities (`src/utils/factor_integrity.py`).

## How to Cite

Chitnis, N. (2025). Bank Disclosure Opacity and Market Performance (ACCT 445). https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase

## License

MIT License (see `LICENSE`).

## Contact

- Project: GitHub Issues on this repository
- Live site: https://nirvanchitnis-cmyk.github.io/ACCT445-Showcase/
