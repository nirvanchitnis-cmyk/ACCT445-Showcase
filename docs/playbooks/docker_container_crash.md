# Playbook: Docker Container Crash

## Symptoms
- `docker ps` shows `Exited` status for `acct445-app` or `acct445-runner`
- Compose health check fails repeatedly
- Kubernetes/Swarm restarts pods every few minutes
- No dashboard at `http://HOST:8501`

## Diagnosis
1. Inspect container logs:
   ```bash
   docker logs acct445-app --tail 200
   docker logs acct445-runner --tail 200
   ```
2. Run containers interactively:
   ```bash
   docker run -it acct445-showcase:latest /bin/bash
   ```
3. Validate mounted volumes:
   ```bash
   ls -R data/ results/ logs/ config/
   ```
4. Ensure images are up to date:
   ```bash
   docker compose build --pull
   docker image ls acct445-showcase
   ```

## Resolution
- Fix missing dependencies by reinstalling requirements, then rebuild images
- Remove corrupted caches: `rm -rf data/cache/*`, rerun `dvc pull`
- Confirm host path permissions allow Docker to write `logs/` and `results/`
- If runner crashes due to schedule/timezone issues, set `TZ=UTC` environment variable

## Prevention
- Enable Docker restart policies (already `unless-stopped` in compose)
- Ship logs to external aggregation (CloudWatch, ELK) to keep history
- Automate nightly `docker compose pull && docker compose up -d --build`
