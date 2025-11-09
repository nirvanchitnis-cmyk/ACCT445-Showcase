# Deployment Guide

Production checklist for ACCT445 Showcase. Follow the stages below to provision
infrastructure, configure secrets, and validate the system end-to-end.

---

## 1. Prerequisites

| Requirement | Notes |
| ----------- | ----- |
| Docker ≥ 24 | Needed for `Dockerfile` / `docker-compose.yml` |
| Docker Compose ≥ 2 | Manages dashboard + runner services |
| Git + GitHub access | Clone repo, manage pull requests, CI |
| Python 3.11 | Local debugging outside containers |
| `dvc>=3` + remote storage | Pull/push data artifacts (`config/sample_cnoi.csv`, `results/*.csv`, `data/cache/`) |
| `CODECOV_TOKEN` (optional) | Enables coverage uploads in CI |

Recommended cloud targets:

* **Single VM** (e.g., AWS EC2, GCP Compute Engine) with Docker + cron
* **Container orchestrator** (ECS, GKE, AKS) for managed restarts and logging

---

## 2. Quick Start (Single Host)

```bash
git clone https://github.com/YOUR_ORG/ACCT445-Showcase.git
cd ACCT445-Showcase

# Install dependencies for local testing
pip install -r requirements.txt

# Pull datasets managed by DVC
dvc pull

# (Optional) run the automated runner once
python src/runner/daily_backtest.py
```

### Build and Run Containers

```bash
docker compose build            # builds acct445-app + backtest-runner
docker compose up -d            # starts both services
docker compose ps               # confirm healthy containers
docker compose logs -f          # tail combined logs
```

`acct445-app` exposes the Streamlit dashboard on port 8501.

### Stop Services

```bash
docker compose down
```

Use `docker compose down -v` to remove named volumes if you no longer need local
cache/result data inside the container.

---

## 3. Configuration

Centralized in `config/config.toml` and accessible via
`src/utils/config.get_config_value`. Key sections:

* `[data]` — paths for the CNOI dataset, cache directories, and results output
* `[backtest]` — number of deciles, weighting scheme, forward-return horizons
* `[transaction_costs]` — spread/impact coefficients for `apply_transaction_costs`
* `[runner]` — `schedule_time`, lookback window, anomaly thresholds
* `[alerts]` — toggle email alerts and provide SMTP placeholders (implementation hooks reside in `src/runner/alerts.py`)

Mount `config/` into the container (done by `docker-compose.yml`) so overrides
apply without rebuilding images. For Kubernetes, project these settings through
ConfigMaps/Secrets and point the environment variable
`ACCT445_CONFIG_PATH=/app/config/config.toml` if relocated.

---

## 4. Monitoring & Observability

| Signal | Location |
| ------ | -------- |
| Runner logs | `logs/runner.log` (also accessible via `docker logs acct445-runner`) |
| Dashboard logs | `docker logs acct445-app` |
| Results health | `results/decile_summary_latest.csv`, `results/decile_long_short_latest.csv` |
| Dashboard | `http://HOSTNAME:8501` |
| Alerts | Logged through `src/runner/alerts.send_alert` (email hooks ready once SMTP configured) |

Health checks:

* `docker-compose.yml` uses HTTP health checks for the dashboard.
* Add cron or external uptime monitors to hit `/` on port 8501 and a custom
  `/health` endpoint (future API Task 5.7) if desired.

---

## 5. Troubleshooting

| Symptom | Resolution |
| ------- | ---------- |
| `dvc pull` fails | Ensure remote credentials; run `dvc remote list` and `dvc remote modify` with proper auth |
| Runner exits immediately | Check `logs/runner.log`, confirm `config/data.cnoi_file` path exists inside container |
| Dashboard shows "No data" warnings | Run `python src/runner/daily_backtest.py` or populate mock data via `python scripts/create_mock_results.py --force` |
| Docker health check unhealthy | Tail `docker logs acct445-app`; confirm results CSVs exist and Streamlit can read them |
| CI docs build fails | Run `make -C docs clean html` locally to inspect errors (missing imports, etc.) |

See `docs/playbooks/` for deeper incident workflows covering data-quality
degradation, backtest failures, performance regressions, and container crashes.

---

## 6. Production Hardening Tips

* Configure a DVC remote (S3/GCS/Azure) and export credentials to CI before publishing artifacts.
* Attach persistent storage (EBS/EFS) to keep `data/cache` and `results/` between deployments.
* Route dashboard traffic through HTTPS (e.g., Nginx reverse proxy or AWS ALB).
* Integrate alerts with SMTP, Slack, or PagerDuty by extending `src/runner/alerts.py`.
* Add a `CODECOV_TOKEN` secret to GitHub for richer coverage reporting on PRs.
