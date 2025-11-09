"""
Factor data integrity checking using SHA-256 checksums.

Ensures factor data hasn't been corrupted or tampered with, preventing
silent factor mis-alignment that can create phantom alphas.

References:
- Ken French Data Library: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


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


def generate_manifest(
    factor_files: dict[str, Path | str],
    output_path: Path | str = Path("config/factors_manifest.json"),
) -> None:
    """
    Generate a factor manifest JSON with SHA-256 checksums.

    Args:
        factor_files: Dict of {factor_name: file_path}
        output_path: Where to save manifest

    Example:
        >>> generate_manifest({
        ...     "ff5_daily": "data/factors/F-F_Research_Data_5_Factors_2x3_daily.CSV",
        ...     "mom_daily": "data/factors/F-F_Momentum_Factor_daily.CSV"
        ... })
    """
    manifest = {}
    for name, path in factor_files.items():
        path = Path(path)
        if not path.exists():
            logger.warning("Skipping %s - file not found: %s", name, path)
            continue

        sha = sha256_file(path)
        manifest[name] = {"path": str(path), "sha256": sha}
        logger.info("Added %s: %s (SHA-256: %s)", name, path, sha[:16] + "...")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Factor manifest saved to %s (%d files)", output_path, len(manifest))


if __name__ == "__main__":
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
