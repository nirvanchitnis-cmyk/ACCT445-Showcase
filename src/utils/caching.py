"""Utilities for simple disk-based caching."""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def _default_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Return a stable hash for arbitrary call arguments."""
    payload = pickle.dumps((args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.md5(payload).hexdigest()


def disk_cache(
    cache_dir: str | Path,
    *,
    key_func: Callable[[tuple[Any, ...], dict[str, Any]], str] | None = None,
    disable_cache_kwarg: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Persist function outputs to disk keyed by arguments.

    Args:
        cache_dir: Directory where cache files are stored.
        key_func: Optional custom function that returns a cache key string.
        disable_cache_kwarg: When supplied, a truthy value for this keyword skips
            cache reads/writes (useful for forcing refreshes in notebooks).
    """

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if disable_cache_kwarg:
                # Remove the kwarg so it does not affect hashes.
                skip_cache = bool(kwargs.pop(disable_cache_kwarg, False))
            else:
                skip_cache = False

            if skip_cache:
                logger.debug("Cache disabled for %s; executing function.", func.__name__)
                return func(*args, **kwargs)

            key = key_func(args, kwargs) if key_func else _default_key(args, kwargs)
            cache_file = cache_path / f"{func.__name__}_{key}.pkl"

            if cache_file.exists():
                try:
                    with cache_file.open("rb") as handle:
                        logger.debug("Cache hit for %s (%s).", func.__name__, cache_file.name)
                        return pickle.load(handle)
                except (OSError, pickle.PickleError) as exc:
                    logger.warning("Failed reading cache %s: %s", cache_file, exc)
                    cache_file.unlink(missing_ok=True)

            result = func(*args, **kwargs)

            try:
                with cache_file.open("wb") as handle:
                    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.debug("Cached result for %s at %s.", func.__name__, cache_file)
            except (OSError, pickle.PickleError) as exc:
                logger.warning("Unable to cache result for %s: %s", func.__name__, exc)

            return result

        return wrapper

    return decorator


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Stable hash for a dataframe that is insensitive to row order.

    Useful for constructing cache keys without storing the full dataframe bytes.
    """

    if df.empty:
        return "empty_df"

    normalized = df.sort_index(axis=1)
    hashed = pd.util.hash_pandas_object(normalized, index=False).to_numpy()
    hashed.sort()
    return hashlib.md5(hashed.tobytes()).hexdigest()


__all__ = ["disk_cache", "hash_dataframe"]
