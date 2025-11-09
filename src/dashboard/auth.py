"""
Streamlit authentication using streamlit-authenticator.

Default credentials:
- Username: admin
- Password: acct445_demo (change in production!)
"""

from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
import yaml


def load_auth_config(config_path: Path = Path("config/auth.yaml")) -> dict:
    """Load authentication config from YAML."""
    if not config_path.exists():
        # Create default config with hashed password
        hasher = stauth.Hasher()
        hashed_password = hasher.hash("acct445_demo")

        default_config = {
            "credentials": {
                "usernames": {
                    "admin": {"name": "Admin User", "password": hashed_password}  # Bcrypt hash
                }
            },
            "cookie": {
                "name": "acct445_auth_cookie",
                "key": "acct445_secret_key_change_me",  # Change in production!
                "expiry_days": 30,
            },
            "preauthorized": {"emails": []},
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(default_config, f)
        return default_config

    with open(config_path) as f:
        return yaml.safe_load(f)


def authenticate() -> tuple[str, bool, str]:
    """
    Add authentication to Streamlit app.

    Returns:
        (name, authentication_status, username)

    Usage in app.py:
        name, auth_status, username = authenticate()
        if not auth_status:
            st.stop()
    """
    config = load_auth_config()

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )

    name, authentication_status, username = authenticator.login("Login", "main")

    if not authentication_status:
        st.error("Username/password is incorrect")
    elif authentication_status is None:
        st.warning("Please enter your username and password")

    return name, authentication_status, username


def add_logout_button(authenticator):
    """Add logout button to sidebar."""
    st.sidebar.markdown("---")
    authenticator.logout("Logout", "sidebar")
