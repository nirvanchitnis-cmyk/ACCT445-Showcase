"""Tests for factor data integrity checking."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.utils.factor_integrity import generate_manifest, sha256_file, verify_factors


@pytest.fixture
def temp_factor_files():
    """Create temporary factor files for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock factor files
        ff5_path = tmpdir / "ff5.csv"
        mom_path = tmpdir / "mom.csv"

        ff5_path.write_text(
            "Date,Mkt-RF,SMB,HML,RMW,CMA,RF\n2024-01-01,0.01,0.002,0.001,0.001,0.001,0.0001\n"
        )
        mom_path.write_text("Date,Mom\n2024-01-01,0.003\n")

        yield {"ff5": ff5_path, "mom": mom_path, "tmpdir": tmpdir}


class TestSHA256File:
    """Tests for sha256_file()."""

    def test_computes_sha256(self, temp_factor_files):
        """Test SHA-256 computation."""
        ff5_path = temp_factor_files["ff5"]
        sha = sha256_file(ff5_path)

        # Should return 64-character hex string
        assert isinstance(sha, str)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_same_file_same_hash(self, temp_factor_files):
        """Test deterministic hashing."""
        ff5_path = temp_factor_files["ff5"]
        sha1 = sha256_file(ff5_path)
        sha2 = sha256_file(ff5_path)

        assert sha1 == sha2

    def test_different_content_different_hash(self, temp_factor_files):
        """Test different files have different hashes."""
        ff5_sha = sha256_file(temp_factor_files["ff5"])
        mom_sha = sha256_file(temp_factor_files["mom"])

        assert ff5_sha != mom_sha

    def test_raises_on_missing_file(self):
        """Test raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            sha256_file("nonexistent.csv")


class TestGenerateManifest:
    """Tests for generate_manifest()."""

    def test_creates_manifest(self, temp_factor_files):
        """Test manifest generation."""
        tmpdir = temp_factor_files["tmpdir"]
        manifest_path = tmpdir / "manifest.json"

        factor_files = {
            "ff5_daily": temp_factor_files["ff5"],
            "mom_daily": temp_factor_files["mom"],
        }

        generate_manifest(factor_files, output_path=manifest_path)

        # Verify manifest exists
        assert manifest_path.exists()

        # Verify manifest structure
        manifest = json.loads(manifest_path.read_text())
        assert "ff5_daily" in manifest
        assert "mom_daily" in manifest
        assert "path" in manifest["ff5_daily"]
        assert "sha256" in manifest["ff5_daily"]
        assert len(manifest["ff5_daily"]["sha256"]) == 64

    def test_skips_missing_files(self, temp_factor_files):
        """Test skips files that don't exist."""
        tmpdir = temp_factor_files["tmpdir"]
        manifest_path = tmpdir / "manifest.json"

        factor_files = {
            "ff5_daily": temp_factor_files["ff5"],
            "missing": tmpdir / "does_not_exist.csv",  # Missing file
        }

        # Should not raise, just skip the missing file
        generate_manifest(factor_files, output_path=manifest_path)

        manifest = json.loads(manifest_path.read_text())
        assert "ff5_daily" in manifest
        assert "missing" not in manifest  # Skipped


class TestVerifyFactors:
    """Tests for verify_factors()."""

    def test_verifies_valid_factors(self, temp_factor_files):
        """Test successful verification."""
        tmpdir = temp_factor_files["tmpdir"]
        manifest_path = tmpdir / "manifest.json"

        # Generate manifest
        factor_files = {"ff5_daily": temp_factor_files["ff5"]}
        generate_manifest(factor_files, output_path=manifest_path)

        # Verify
        result = verify_factors(manifest_path)

        assert result["all_valid"] is True
        assert "ff5_daily" in result["results"]
        assert result["results"]["ff5_daily"]["status"] == "OK"

    def test_detects_checksum_mismatch(self, temp_factor_files):
        """Test detects when file content changes."""
        tmpdir = temp_factor_files["tmpdir"]
        manifest_path = tmpdir / "manifest.json"
        ff5_path = temp_factor_files["ff5"]

        # Generate manifest
        factor_files = {"ff5_daily": ff5_path}
        generate_manifest(factor_files, output_path=manifest_path)

        # Modify file content
        ff5_path.write_text("Modified content\n")

        # Verification should fail
        with pytest.raises(RuntimeError, match="Factor integrity check FAILED"):
            verify_factors(manifest_path)

    def test_detects_missing_file(self, temp_factor_files):
        """Test detects when factor file is missing."""
        tmpdir = temp_factor_files["tmpdir"]
        manifest_path = tmpdir / "manifest.json"
        ff5_path = temp_factor_files["ff5"]

        # Generate manifest
        factor_files = {"ff5_daily": ff5_path}
        generate_manifest(factor_files, output_path=manifest_path)

        # Delete file
        ff5_path.unlink()

        # Verification should fail
        with pytest.raises(RuntimeError, match="Factor integrity check FAILED"):
            verify_factors(manifest_path)

    def test_handles_missing_manifest(self, temp_factor_files):
        """Test gracefully handles missing manifest."""
        tmpdir = temp_factor_files["tmpdir"]
        nonexistent_manifest = tmpdir / "does_not_exist.json"

        # Should not raise, just warn
        result = verify_factors(nonexistent_manifest)

        assert result["all_valid"] is None
        assert "warning" in result
        assert result["warning"] == "Manifest not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
