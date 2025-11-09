"""Tests for src/utils/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import config as config_module


def test_load_config_reads_toml():
    config_module.load_config.cache_clear()
    data = config_module.load_config()
    assert data["general"]["random_seed"] == 42
    assert "backtest" in data


def test_get_config_value_nested_key():
    config_module.load_config.cache_clear()
    assert config_module.get_config_value("backtest.n_deciles") == 10
    assert config_module.get_config_value("market_data.start_date") == "2023-01-01"


def test_get_config_value_default_when_missing():
    config_module.load_config.cache_clear()
    assert config_module.get_config_value("missing.section", default="fallback") == "fallback"


def test_load_config_missing_file_raises(tmp_path: Path):
    config_module.load_config.cache_clear()
    missing = tmp_path / "nope.toml"
    with pytest.raises(FileNotFoundError):
        config_module.load_config(missing)


def test_get_config_value_handles_non_dict_path():
    config_module.load_config.cache_clear()
    assert (
        config_module.get_config_value("backtest.n_deciles.value", default="fallback") == "fallback"
    )
