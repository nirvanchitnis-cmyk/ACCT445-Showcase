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

## Key Findings

### Main Result
**Banks with opaque CECL disclosures underperform transparent banks by 220 basis points per quarter (8.8% annualized alpha)**

This opacity premium is:
- **Statistically significant:** t = 3.45, p < 0.001 (passes Harvey-Liu-Zhu multiple testing threshold)
- **Risk-adjusted:** Alpha survives Fama-French 5-factor + Momentum controls
- **Economically large:** Long-short portfolio earns 2.2% quarterly (identical raw return and FF5 alpha)
- **Causal:** Difference-in-differences with 2-way clustering (β = -4.8%, t = -3.20, p = 0.001)
- **Crisis-relevant:** Opaque banks suffered 10.5 pp worse CAR during SVB collapse (t = 3.42, p < 0.001)

### Specific Results
| Test | Estimate | t-stat | p-value | Standard Errors |
|------|----------|--------|---------|-----------------|
| Decile Backtest (Long-Short) | 2.2% per quarter | 3.18 | 0.002 | Newey-West (3 lags) |
| FF5 Alpha | 2.2% per quarter | 3.45 | <0.001 | Newey-West (3 lags) |
| Carhart Alpha (FF5+Mom) | 1.9% per quarter | 3.12 | 0.002 | Newey-West (3 lags) |
| Panel Regression (CNOI coef) | -8.2 bps per 1-pt CNOI | -3.15 | 0.002 | Driscoll-Kraay |
| DiD (Treat × Post) | -4.8% per quarter | -3.20 | 0.001 | 2-way clustered |
| SVB Event Study (Q4-Q1 CAR) | -10.5 pp | 3.42 | <0.001 | BMP, Corrado, Sign |

### CNOI Dimension Importance
1. **Stability (S):** Explains 46% of CNOI variance, ρ = 0.42 with volatility
2. **Required Items (R):** Explains 37% of variance, ρ = 0.31 with volatility
3. **Consistency (X):** Explains 27% of variance, ρ = 0.25 with volatility

**Interpretation:** Period-over-period stability drives investor uncertainty more than static readability.

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

**APA:**
```
Chitnis, N. (2025). Bank disclosure opacity and market performance: Evidence from CECL notes.
    ACCT 445 Research Project. Retrieved from https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase
```

**BibTeX:**
```bibtex
@misc{chitnis2025cecl,
  author = {Chitnis, Nirvan},
  title = {Bank Disclosure Opacity and Market Performance: Evidence from {CECL} Notes},
  year = {2025},
  howpublished = {\url{https://github.com/nirvanchitnis-cmyk/ACCT445-Showcase}},
  note = {ACCT 445 Research Project}
}
```

**JEL Classification:** G12 (Asset Pricing), G14 (Information and Market Efficiency), M41 (Accounting)

## License

MIT License (see `LICENSE`).

## Contact

- Project: GitHub Issues on this repository
- Live site: https://nirvanchitnis-cmyk.github.io/ACCT445-Showcase/
