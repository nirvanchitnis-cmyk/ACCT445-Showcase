import os
import time

from fastapi.testclient import TestClient

from src.infra import health_server


def _touch_results_file(tmp_path, hours_old: float) -> str:
    """Create a CSV file whose modified time is `hours_old` hours in the past."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "decile_summary.csv"
    csv_path.write_text("ticker,ret\nABC,0.01\n")
    timestamp = time.time() - hours_old * 3600
    os.utime(csv_path, (timestamp, timestamp))
    return str(results_dir)


def _make_client(monkeypatch, sha: str = "abc123"):
    """Return a TestClient with git SHA patched for deterministic responses."""
    monkeypatch.setenv("PYTHON_VERSION", "3.11-test")
    monkeypatch.setattr(health_server.subprocess, "check_output", lambda *_, **__: f"{sha}\n")
    return TestClient(health_server.app)


def test_healthz_reports_healthy_for_recent_results(tmp_path, monkeypatch):
    results_dir = _touch_results_file(tmp_path, hours_old=2)
    monkeypatch.setenv("RESULTS_DIR", results_dir)

    client = _make_client(monkeypatch, sha="deadbeef")
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["git_sha"] == "deadbeef"
    assert payload["checks"] == {
        "data_exists": True,
        "data_fresh": True,
        "data_recent": True,
    }


def test_healthz_degraded_when_data_is_stale(tmp_path, monkeypatch):
    results_dir = _touch_results_file(tmp_path, hours_old=72)
    monkeypatch.setenv("RESULTS_DIR", results_dir)

    client = _make_client(monkeypatch)
    response = client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["checks"]["data_exists"] is True
    assert payload["checks"]["data_recent"] is True
    assert payload["checks"]["data_fresh"] is False


def test_healthz_unhealthy_when_results_missing(tmp_path, monkeypatch):
    missing_dir = tmp_path / "no_results_here"
    # Intentionally do not create the directory to hit the early-return branch
    monkeypatch.setenv("RESULTS_DIR", str(missing_dir))

    client = _make_client(monkeypatch)
    response = client.get("/healthz")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "unhealthy"
    assert payload["checks"]["data_exists"] is False


def test_healthz_unhealthy_when_directory_empty(tmp_path, monkeypatch):
    empty_dir = tmp_path / "empty_results"
    empty_dir.mkdir()
    monkeypatch.setenv("RESULTS_DIR", str(empty_dir))

    client = _make_client(monkeypatch)
    response = client.get("/healthz")
    assert response.status_code == 503
    payload = response.json()
    assert payload["checks"]["data_exists"] is False


def test_ping_endpoint_returns_timestamp(monkeypatch):
    client = _make_client(monkeypatch)
    response = client.get("/ping")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pong"
    assert "timestamp" in payload


def test_root_endpoint_returns_metadata(monkeypatch):
    client = _make_client(monkeypatch)
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "online"
    assert payload["service"] == "ACCT445-Showcase Health Check"


def test_get_git_sha_gracefully_handles_errors(monkeypatch):
    def _raise(*_, **__):
        raise RuntimeError("git unavailable")

    monkeypatch.setattr(health_server.subprocess, "check_output", _raise)
    assert health_server.get_git_sha() == "unknown"
