"""Robust SEC EDGAR API client with retry logic and caching."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

from src.utils.exceptions import ExternalAPIError
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "sec"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_HOURS = 24.0

# SEC requires descriptive User-Agent headers
# https://www.sec.gov/os/accessing-edgar-data
USER_AGENTS = (
    "ACCT445-Showcase Research contact@university.edu",
    "ACCT445-Showcase Backup student@university.edu",
)


def _load_cache(cache_file: Path) -> dict[str, dict[str, str]]:
    """Load cached SEC data, raising if the cache is unreadable."""
    try:
        with cache_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - corruption is rare
        logger.warning("Cached SEC mapping corrupted: %s", exc)
        raise


def fetch_sec_ticker_mapping(
    use_cache: bool = True, max_retries: int = 3
) -> dict[str, dict[str, str]]:
    """
    Fetch SEC's company ticker mapping with retries and caching.

    Args:
        use_cache: If True, use cached data when available and fresh (<24h)
        max_retries: Maximum number of attempts before failing

    Returns:
        Dict keyed by zero-padded CIK (str) with ticker/title metadata

    Raises:
        ExternalAPIError: When data cannot be retrieved
    """

    cache_file = CACHE_DIR / "company_tickers.json"
    if use_cache and cache_file.exists():
        cache_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if cache_age_hours < CACHE_TTL_HOURS:
            try:
                cached = _load_cache(cache_file)
                logger.info("Loading SEC mapping from cache (age: %.1fh)", cache_age_hours)
                return cached
            except json.JSONDecodeError:
                logger.warning("Ignoring corrupted cache at %s", cache_file)

    if max_retries < 1:
        raise ExternalAPIError("max_retries must be at least 1")

    url = "https://www.sec.gov/files/company_tickers.json"

    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": USER_AGENTS[attempt % len(USER_AGENTS)],
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            }
            logger.info("Fetching SEC mapping (attempt %s/%s)", attempt + 1, max_retries)
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            payload = response.json()
            mapping: dict[str, dict[str, str]] = {}
            for entry in payload.values():
                cik = str(entry["cik_str"]).zfill(10)
                mapping[cik] = {
                    "ticker": entry["ticker"],
                    "title": entry["title"],
                }

            with cache_file.open("w", encoding="utf-8") as handle:
                json.dump(mapping, handle)

            logger.info("✓ SEC mapping fetched: %s companies", len(mapping))
            return mapping

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 403 and attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning(
                    "SEC API 403 Forbidden (attempt %s). Retrying in %ss.", attempt + 1, wait
                )
                time.sleep(wait)
                continue
            raise ExternalAPIError("SEC API HTTP error") from exc
        except requests.RequestException as exc:
            if attempt < max_retries - 1:
                wait = 2**attempt
                logger.warning("SEC API request error (%s). Retrying in %ss.", exc, wait)
                time.sleep(wait)
                continue
            raise ExternalAPIError("Failed to fetch SEC mapping") from exc
        except json.JSONDecodeError as exc:
            raise ExternalAPIError("SEC API returned invalid JSON") from exc

    raise ExternalAPIError("Failed to fetch SEC mapping after all retries")
