"""Lightweight validation for Docker assets."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def read_repo_file(relative_path: str) -> str:
    file_path = REPO_ROOT / relative_path
    assert file_path.exists(), f"{relative_path} does not exist"
    return file_path.read_text(encoding="utf-8")


def test_dockerfile_base_image_and_ports() -> None:
    dockerfile = read_repo_file("Dockerfile")
    assert "FROM python:3.11-slim" in dockerfile
    assert "EXPOSE 8501" in dockerfile
    assert "EXPOSE 8000" in dockerfile


def test_dockerfile_multi_stage_build() -> None:
    """Verify multi-stage build is properly configured."""
    dockerfile = read_repo_file("Dockerfile")
    assert "AS builder" in dockerfile, "Multi-stage build not found"
    assert "Stage 1: Builder" in dockerfile
    assert "Stage 2: Runtime" in dockerfile


def test_dockerfile_non_root_user() -> None:
    """Verify non-root user is created with UID 1000."""
    dockerfile = read_repo_file("Dockerfile")
    assert "useradd -m -u 1000 appuser" in dockerfile
    assert "USER appuser" in dockerfile


def test_dockerfile_security_best_practices() -> None:
    """Verify security best practices are followed."""
    dockerfile = read_repo_file("Dockerfile")
    assert "--no-install-recommends" in dockerfile, "Should minimize installed packages"
    assert "rm -rf /var/lib/apt/lists" in dockerfile, "Should clean apt cache"
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile, "Should disable .pyc files"


def test_docker_compose_services_defined() -> None:
    compose = read_repo_file("docker-compose.yml")
    assert "acct445-dashboard" in compose
    assert "backtest-runner" in compose
    assert "./data:/app/data" in compose


def test_docker_compose_timezone_handling() -> None:
    """Verify timezone is set for DST-safe scheduling."""
    compose = read_repo_file("docker-compose.yml")
    assert "TZ=America/New_York" in compose


def test_docker_compose_healthchecks() -> None:
    """Verify both services have healthchecks."""
    compose = read_repo_file("docker-compose.yml")
    assert compose.count("healthcheck:") >= 2


def test_docker_compose_named_volumes() -> None:
    """Verify named volumes for job locks."""
    compose = read_repo_file("docker-compose.yml")
    assert "runner-locks:" in compose
    assert "volumes:" in compose


def test_dockerignore_has_key_patterns() -> None:
    dockerignore = read_repo_file(".dockerignore")
    required = {"__pycache__", ".git/", ".pytest_cache/", ".dvc/tmp/", ".vscode/"}
    for pattern in required:
        assert pattern in dockerignore, f"Missing pattern: {pattern}"


def test_runner_scheduler_exists() -> None:
    """Verify production scheduler exists."""
    scheduler = REPO_ROOT / "src" / "runner" / "scheduler.py"
    assert scheduler.exists(), "Production scheduler required for docker-compose"


def test_runner_placeholder_exists() -> None:
    runner = REPO_ROOT / "src" / "runner" / "daily_backtest.py"
    assert runner.exists(), "Runner placeholder required for docker-compose"
