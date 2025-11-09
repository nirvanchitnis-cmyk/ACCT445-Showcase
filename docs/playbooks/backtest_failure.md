# Playbook: Backtest Failure

## Symptoms
- Runner exits with "Daily update failed" alert
- `results/decile_summary_latest.csv` missing or zero byte
- CI `pytest` fails in `tests/test_runner.py` or `tests/test_decile_backtest.py`
- Docker logs show stack traces referencing `run_decile_backtest`

## Diagnosis
1. Tail runner logs:
   ```bash
   tail -n 200 logs/runner.log
   ```
2. Reproduce locally with verbose logging:
   ```bash
   python src/runner/daily_backtest.py --debug
   ```
   (Set `LOG_LEVEL=DEBUG` in environment if needed.)
3. Validate dataset inputs:
   - Check `config/config.toml` paths
   - Inspect `results/cnoi_with_tickers.csv`
   - Confirm market data exists for all tickers/date ranges
4. Run targeted tests:
   ```bash
   pytest tests/test_decile_backtest.py -k run_decile_backtest
   pytest tests/test_runner.py -k run_daily_update
   ```

## Resolution
- Fix schema issues (missing columns, wrong dtypes) before calling `run_decile_backtest`
- Reduce missing tickers by refreshing mapping (see data-quality playbook)
- Increase lookback window in `config` if market data range too short
- Ensure transaction-cost config is numeric; invalid values propagate to net returns

## Prevention
- Keep `config/config.toml` checked in and code-reviewed for every change
- Use DVC to version `results/decile_summary_latest.csv` so regressions are diffable
- Extend tests with new fixtures whenever analytics modules change
