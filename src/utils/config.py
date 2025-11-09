"""TOML-based configuration loader."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import toml

CONFIG_FILE = Path(__file__).resolve().parents[2] / "config" / "config.toml"


@lru_cache(maxsize=1)
def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and cache the project configuration."""

    path = config_path or CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    return toml.load(path)


def get_config_value(
    key_path: str,
    default: Any | None = None,
    *,
    config: dict[str, Any] | None = None,
) -> Any:
    """Retrieve a configuration value using dot-notation."""

    data = config or load_config()
    value: Any = data
    for key in key_path.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
        if value is default:
            break
    return value
