# Playbook: Data Quality Degradation

## Symptoms
- Dashboard "Data Quality" page shows coverage < 90%
- `send_alert` message: "CNOI signal weakened" or "Missing tickers"
- Runner logs contain warnings about empty merge results
- Recent `results/cnoi_with_tickers.csv` has NaN tickers or duplicated CIKs

## Diagnosis
1. Inspect runner logs:
   ```bash
   tail -n 200 logs/runner.log
   ```
2. Validate CNOI schema + ticker enrichment:
   ```bash
   python - <<'PY'
   import pandas as pd
   from src.data.cik_ticker_mapper import enrich_cnoi_with_tickers
   df = pd.read_csv("config/sample_cnoi.csv", parse_dates=["filing_date"])
   enriched = enrich_cnoi_with_tickers(df)
   print(enriched["ticker"].isna().mean())
   PY
   ```
3. Confirm SEC mapping freshness:
   ```bash
   python -m src.data.cik_ticker_mapper --refresh
   ```
4. Check DVC status to ensure `config/sample_cnoi.csv` matches expectations:
   ```bash
   dvc status
   ```

## Resolution
- Refresh the CIK→ticker mapping (`--refresh` flag) and rerun the runner
- Re-run `scripts/create_mock_results.py --force` if testing scenarios without live filings
- If filings are missing columns, regenerate `results/cnoi_with_tickers.csv` via
  Notebook 01 (Data Exploration) or update the upstream ETL

## Prevention
- Schedule weekly SEC ticker refreshes via cron/CI
- Add a CI check that fails when coverage < configured threshold
- Expand `config/cik_ticker_overrides.csv` for known problematic issuers
