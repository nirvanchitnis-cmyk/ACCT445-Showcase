"""
Lightweight FastAPI health check server for monitoring and orchestration.

Exposes `/healthz` endpoint for:
- Docker healthchecks (HEALTHCHECK directive)
- Kubernetes liveness/readiness probes
- External monitoring (Datadog, Prometheus, etc.)

Runs separately from Streamlit dashboard to provide system-level status
even if the main app is down or authentication is failing.

Usage:
    # Run standalone
    python -m uvicorn src.infra.health_server:app --host 0.0.0.0 --port 8080

    # In docker-compose.yml
    services:
      health:
        command: uvicorn src.infra.health_server:app --host 0.0.0.0 --port 8080
        healthcheck:
          test: ["CMD", "curl", "-f", "http://localhost:8080/healthz"]
          interval: 30s
          timeout: 10s
          retries: 3
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="ACCT445 Health Check", version="1.0.0")


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def get_data_age_hours() -> float:
    """Get age of most recent results file in hours."""
    results_dir = Path(os.environ.get("RESULTS_DIR", "results"))
    if not results_dir.exists():
        return 999.0

    csv_files = list(results_dir.glob("*.csv"))
    if not csv_files:
        return 999.0

    # Find most recent file
    most_recent = max(f.stat().st_mtime for f in csv_files)
    age_seconds = datetime.now(timezone.utc).timestamp() - most_recent
    return age_seconds / 3600


@app.get("/")
def root():
    """Root endpoint with basic info."""
    return {"service": "ACCT445-Showcase Health Check", "status": "online", "version": "1.0.0"}


@app.get("/healthz")
def healthz():
    """
    Health check endpoint for monitoring.

    Returns:
        JSON with status (healthy/degraded/unhealthy), system info, and data freshness

    Status logic:
        - healthy: All checks pass, data <48h old
        - degraded: System OK but data stale (>48h)
        - unhealthy: Critical failures (missing data, old data >168h)

    Example response:
        {
            "status": "healthy",
            "timestamp": "2024-01-01T12:00:00Z",
            "git_sha": "a1b2c3d",
            "python_version": "3.11.7",
            "data_age_hours": 12.5,
            "checks": {
                "data_fresh": true,
                "data_exists": true
            }
        }
    """
    data_age_hours = get_data_age_hours()

    checks = {
        "data_exists": data_age_hours < 999,  # Files found
        "data_fresh": data_age_hours < 48,  # Updated in last 48 hours
        "data_recent": data_age_hours < 168,  # Updated in last week
    }

    # Determine overall status
    if all(checks.values()):
        status = "healthy"
    elif checks["data_exists"] and checks["data_recent"]:
        status = "degraded"
    else:
        status = "unhealthy"

    response = {
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
        "data_age_hours": round(data_age_hours, 2),
        "checks": checks,
    }

    # Return 200 if healthy/degraded, 503 if unhealthy
    status_code = 200 if status != "unhealthy" else 503

    return JSONResponse(content=response, status_code=status_code)


@app.get("/ping")
def ping():
    """Simple ping endpoint for basic connectivity check."""
    return {"status": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    import uvicorn

    print("Starting ACCT445 Health Check Server")
    print("=" * 50)
    print("Endpoints:")
    print("  GET /healthz  - Full health check")
    print("  GET /ping     - Simple connectivity test")
    print("  GET /         - Service info")
    print()
    uvicorn.run(app, host="0.0.0.0", port=8080)
