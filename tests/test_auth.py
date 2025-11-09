"""Tests for Streamlit authentication module."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.dashboard.auth import authenticate, load_auth_config


@pytest.fixture
def temp_auth_config():
    """Provide a temporary auth config file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config = {
            "credentials": {
                "usernames": {
                    "testuser": {
                        "name": "Test User",
                        "password": "$2b$12$test_hashed_password",
                    }
                }
            },
            "cookie": {
                "name": "test_auth_cookie",
                "key": "test_secret_key",
                "expiry_days": 30,
            },
            "preauthorized": {"emails": []},
        }
        yaml.dump(config, f)
        config_path = Path(f.name)

    yield config_path

    # Cleanup
    if config_path.exists():
        config_path.unlink()


def test_load_auth_config_existing_file(temp_auth_config):
    """Test loading existing auth config file."""
    config = load_auth_config(temp_auth_config)

    assert "credentials" in config
    assert "cookie" in config
    assert "preauthorized" in config
    assert "testuser" in config["credentials"]["usernames"]


def test_load_auth_config_creates_default():
    """Test that default config is created if file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "auth.yaml"
        assert not config_path.exists()

        config = load_auth_config(config_path)

        # Verify default config was created
        assert config_path.exists()
        assert "credentials" in config
        assert "admin" in config["credentials"]["usernames"]
        assert config["cookie"]["name"] == "acct445_auth_cookie"


def test_default_config_has_hashed_password():
    """Test that default config has bcrypt-hashed password."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "auth.yaml"
        config = load_auth_config(config_path)

        admin_password = config["credentials"]["usernames"]["admin"]["password"]
        # Bcrypt hashes start with $2b$ or $2a$
        assert admin_password.startswith("$2")
        # Bcrypt hashes are 60 characters long
        assert len(admin_password) == 60


def test_authenticate_returns_tuple():
    """Test authenticate returns (name, status, username) tuple."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir) / "auth.yaml"

        with patch("src.dashboard.auth.st"):
            with patch("src.dashboard.auth.stauth.Authenticate") as mock_auth_class:
                mock_authenticator = MagicMock()
                mock_authenticator.login.return_value = ("Admin User", True, "admin")
                mock_auth_class.return_value = mock_authenticator

                name, status, username = authenticate()

                assert isinstance(name, str)
                assert isinstance(status, bool)
                assert isinstance(username, str)


def test_authenticate_handles_failed_login():
    """Test authenticate handles failed login (wrong password)."""
    with patch("src.dashboard.auth.st") as mock_st:
        with patch("src.dashboard.auth.stauth.Authenticate") as mock_auth_class:
            mock_authenticator = MagicMock()
            mock_authenticator.login.return_value = (None, False, None)
            mock_auth_class.return_value = mock_authenticator

            name, status, username = authenticate()

            # Should show error message
            mock_st.error.assert_called_once_with("Username/password is incorrect")
            assert status is False


def test_authenticate_handles_no_credentials():
    """Test authenticate handles case when no credentials entered."""
    with patch("src.dashboard.auth.st") as mock_st:
        with patch("src.dashboard.auth.stauth.Authenticate") as mock_auth_class:
            mock_authenticator = MagicMock()
            mock_authenticator.login.return_value = (None, None, None)
            mock_auth_class.return_value = mock_authenticator

            name, status, username = authenticate()

            # Should show warning message
            mock_st.warning.assert_called_once_with("Please enter your username and password")
            assert status is None


def test_config_has_required_fields():
    """Test that config has all required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "auth.yaml"
        config = load_auth_config(config_path)

        # Check credentials structure
        assert "credentials" in config
        assert "usernames" in config["credentials"]
        assert "admin" in config["credentials"]["usernames"]
        assert "name" in config["credentials"]["usernames"]["admin"]
        assert "password" in config["credentials"]["usernames"]["admin"]

        # Check cookie structure
        assert "cookie" in config
        assert "name" in config["cookie"]
        assert "key" in config["cookie"]
        assert "expiry_days" in config["cookie"]

        # Check preauthorized
        assert "preauthorized" in config
        assert "emails" in config["preauthorized"]


def test_cookie_expiry_days_is_valid():
    """Test that cookie expiry is set to reasonable value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "auth.yaml"
        config = load_auth_config(config_path)

        expiry = config["cookie"]["expiry_days"]
        assert isinstance(expiry, int)
        assert 1 <= expiry <= 365  # Between 1 day and 1 year
