"""
Factor data integrity checking using SHA-256 checksums.

Ensures factor data hasn't been corrupted or tampered with, preventing
silent factor mis-alignment that can create phantom alphas.

Critical: Jan 2025 CRSP format switch from FIZ to CIZ affects factor construction.
Track provenance (source, format, fetch date) to ensure replicability.

References:
- Ken French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- CRSP FIZ→CIZ change: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FactorProvenance:
    """
    Metadata for factor data provenance tracking.

    Critical for replicability after Jan 2025 CRSP FIZ→CIZ format switch.

    Attributes:
        source_url: URL where factor data was downloaded
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        crsp_format: CRSP data format ('FIZ' pre-2025, 'CIZ' post-2025)
        fetched_at: ISO8601 timestamp of download
        nyse_breakpoints: True if sorts use NYSE breakpoints (recommended)
        description: Human-readable description
    """

    source_url: str
    frequency: str
    crsp_format: str  # "FIZ" or "CIZ"
    fetched_at: str  # ISO8601
    nyse_breakpoints: bool
    description: str = ""


def sha256_file(path: Path | str) -> str:
    """
    Compute SHA-256 checksum of a file.

    Args:
        path: Path to file

    Returns:
        Hexadecimal digest string

    Example:
        >>> sha256_file("data/factors/F-F_Research_Data_5_Factors_2x3_daily.CSV")
        '3f5a2b1c...'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Factor file not found: {path}")

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_factors(manifest_path: Path | str = Path("config/factors_manifest.json")) -> dict:
    """
    Verify factor data integrity against manifest checksums.

    Args:
        manifest_path: Path to JSON manifest with expected checksums

    Returns:
        {
            'all_valid': bool,
            'results': {factor_name: {'status': 'OK/MISMATCH/MISSING', ...}}
        }

    Raises:
        RuntimeError: If any factor fails checksum validation

    Example:
        >>> verify_factors()  # Raises if checksums don't match
        >>> verify_factors()  # Returns {'all_valid': True, ...} if OK
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        logger.warning(
            "Factor manifest not found at %s. Skipping integrity check (not production-safe).",
            manifest_path,
        )
        return {"all_valid": None, "results": {}, "warning": "Manifest not found"}

    manifest = json.loads(manifest_path.read_text())
    results = {}
    all_valid = True

    for name, spec in manifest.items():
        path = Path(spec["path"])
        expected_sha = spec["sha256"]

        if not path.exists():
            results[name] = {
                "status": "MISSING",
                "expected": expected_sha,
                "actual": None,
                "path": str(path),
            }
            all_valid = False
            logger.error("[FACTOR INTEGRITY] %s MISSING: %s", name, path)
            continue

        actual_sha = sha256_file(path)

        if actual_sha != expected_sha:
            results[name] = {
                "status": "MISMATCH",
                "expected": expected_sha,
                "actual": actual_sha,
                "path": str(path),
            }
            all_valid = False
            logger.error(
                "[FACTOR INTEGRITY] %s MISMATCH!\nExpected: %s\nActual: %s\nPath: %s",
                name,
                expected_sha,
                actual_sha,
                path,
            )
        else:
            results[name] = {
                "status": "OK",
                "expected": expected_sha,
                "actual": actual_sha,
                "path": str(path),
            }
            logger.info("[FACTOR INTEGRITY] %s OK (SHA-256: %s)", name, actual_sha[:16] + "...")

    if not all_valid:
        failed = [name for name, r in results.items() if r["status"] != "OK"]
        raise RuntimeError(
            f"Factor integrity check FAILED for {len(failed)} file(s): {', '.join(failed)}. "
            "This may indicate data corruption or tampering. "
            "Re-download from Ken French Data Library or update manifest if intentional."
        )

    logger.info("[FACTOR INTEGRITY] All %d factor files verified successfully", len(results))
    return {"all_valid": True, "results": results}


def record_provenance(
    factor_file: Path | str,
    source_url: str,
    frequency: str,
    crsp_format: str,
    nyse_breakpoints: bool = True,
    description: str = "",
) -> dict:
    """
    Record provenance metadata for a factor file.

    Args:
        factor_file: Path to factor CSV file
        source_url: Ken French Data Library URL
        frequency: 'daily', 'weekly', or 'monthly'
        crsp_format: 'FIZ' (pre-2025) or 'CIZ' (post-2025)
        nyse_breakpoints: True if sorts use NYSE breakpoints
        description: Human-readable description

    Returns:
        Dict with sha256 and provenance metadata

    Example:
        >>> prov = record_provenance(
        ...     "data/factors/ff5_daily.csv",
        ...     source_url="https://mba.tuck.dartmouth.edu/...",
        ...     frequency="daily",
        ...     crsp_format="CIZ",  # Post-2025 format
        ...     nyse_breakpoints=True
        ... )
    """
    factor_file = Path(factor_file)
    sha = sha256_file(factor_file)

    provenance = FactorProvenance(
        source_url=source_url,
        frequency=frequency,
        crsp_format=crsp_format,
        fetched_at=dt.datetime.utcnow().isoformat(),
        nyse_breakpoints=nyse_breakpoints,
        description=description,
    )

    return {"path": str(factor_file), "sha256": sha, "provenance": asdict(provenance)}


def generate_manifest(
    factor_files: dict[str, Path | str | dict],
    output_path: Path | str = Path("config/factors_manifest.json"),
) -> None:
    """
    Generate a factor manifest JSON with SHA-256 checksums and provenance.

    Args:
        factor_files: Dict of {factor_name: file_path} OR
                      {factor_name: {path: ..., provenance: {...}}}
        output_path: Where to save manifest

    Example:
        >>> generate_manifest({
        ...     "ff5_daily": {
        ...         "path": "data/factors/ff5_daily.csv",
        ...         "source_url": "https://...",
        ...         "frequency": "daily",
        ...         "crsp_format": "CIZ",
        ...         "nyse_breakpoints": True
        ...     }
        ... })
    """
    manifest = {}
    for name, spec in factor_files.items():
        # Handle simple path or full spec dict
        if isinstance(spec, str | Path):
            path = Path(spec)
            if not path.exists():
                logger.warning("Skipping %s - file not found: %s", name, path)
                continue
            entry = {"path": str(path), "sha256": sha256_file(path)}
        else:
            # Full spec with provenance
            path = Path(spec["path"])
            if not path.exists():
                logger.warning("Skipping %s - file not found: %s", name, path)
                continue

            # Record full provenance
            entry = record_provenance(
                path,
                source_url=spec.get("source_url", "unknown"),
                frequency=spec.get("frequency", "unknown"),
                crsp_format=spec.get("crsp_format", "unknown"),
                nyse_breakpoints=spec.get("nyse_breakpoints", True),
                description=spec.get("description", ""),
            )
        manifest[name] = entry

        logger.info("Added %s: %s (SHA-256: %s)", name, path, entry["sha256"][:16] + "...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Factor manifest saved to %s (%d files)", output_path, len(manifest))


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    # Example usage: verify existing factors or generate new manifest
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        # Generate manifest from typical factor files
        factor_files = {
            "ff5_daily": "data/factors/F-F_Research_Data_5_Factors_2x3_daily.CSV",
            "mom_daily": "data/factors/F-F_Momentum_Factor_daily.CSV",
        }
        generate_manifest(factor_files)
    else:
        # Verify existing manifest
        verify_factors()
