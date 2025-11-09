# Playbook: Performance Degradation

## Symptoms
- Runner takes > 15 minutes to complete
- Dashboard loads slowly (plots > 5 s)
- CPU pegged due to sequential market downloads
- Alerts triggered for stale data because schedule misses windows

## Diagnosis
1. Measure runner duration:
   ```bash
   /usr/bin/time -v python src/runner/daily_backtest.py
   ```
2. Inspect caching effectiveness:
   ```bash
   ls -lh .cache/robustness
   ls -lh data/cache/yfinance
   ```
3. Confirm parallel fetch:
   - Ensure `joblib` installed
   - `config/backtest.use_parallel_market_download` (if added) enabled
4. Profile hot spots:
   ```bash
   python -m cProfile -o profile.out src/runner/daily_backtest.py
   snakeviz profile.out
   ```

## Resolution
- Warm the yfinance cache via `python src/utils/market_data.py --prime`
- Use `fetch_bulk_data(..., use_parallel=True)` inside runner if not already default
- Enable disk caching decorator for robustness routines (`use_cache=True`)
- Scale container CPU/memory if resource constrained

## Prevention
- Keep transaction-cost and robustness caches on persistent storage
- Run daily updates at off-peak hours (`runner.schedule_time`)
- Monitor wall-clock runtimes and alert when exceeding SLA
