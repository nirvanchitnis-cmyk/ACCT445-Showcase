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


def test_docker_compose_services_defined() -> None:
    compose = read_repo_file("docker-compose.yml")
    assert "acct445-app" in compose
    assert "backtest-runner" in compose
    assert "./data:/app/data" in compose


def test_dockerignore_has_key_patterns() -> None:
    dockerignore = read_repo_file(".dockerignore")
    required = {"__pycache__", ".git/", ".pytest_cache/", ".dvc/tmp/", ".vscode/"}
    for pattern in required:
        assert pattern in dockerignore, f"Missing pattern: {pattern}"


def test_runner_placeholder_exists() -> None:
    runner = REPO_ROOT / "src" / "runner" / "daily_backtest.py"
    assert runner.exists(), "Runner placeholder required for docker-compose"
