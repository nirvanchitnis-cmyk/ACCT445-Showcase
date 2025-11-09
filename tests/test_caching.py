"""Tests for disk caching utilities."""

from __future__ import annotations

import pandas as pd

from src.utils.caching import disk_cache, hash_dataframe


def test_disk_cache_hits(tmp_path):
    calls = {"count": 0}

    @disk_cache(tmp_path)
    def add(a: int, b: int) -> int:
        calls["count"] += 1
        return a + b

    assert add(1, 2) == 3
    assert add(1, 2) == 3
    assert calls["count"] == 1  # Second call should be cached


def test_disk_cache_disable_kwarg(tmp_path):
    calls = {"count": 0}

    @disk_cache(tmp_path, disable_cache_kwarg="force_refresh")
    def multiply(a: int, b: int, *, force_refresh: bool = False) -> int:
        del force_refresh  # Only used for cache control
        calls["count"] += 1
        return a * b

    assert multiply(2, 3) == 6
    assert multiply(2, 3) == 6  # cached
    assert multiply(2, 3, force_refresh=True) == 6
    assert calls["count"] == 2  # force refresh bypassed cache once


def test_hash_dataframe_order_invariant():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    shuffled = df.sample(frac=1).reset_index(drop=True)
    assert hash_dataframe(df) == hash_dataframe(shuffled)
