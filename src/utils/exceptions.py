"""Custom exception hierarchy for ACCT445 Showcase."""

from __future__ import annotations


class ACCT445Error(Exception):
    """Base class for project specific exceptions."""


class ExternalAPIError(ACCT445Error):
    """Raised when third-party data sources fail."""


class DataValidationError(ACCT445Error):
    """Raised when input data violates schema expectations."""


class DataDownloadError(ACCT445Error):
    """Raised when market data downloads fail or return empty outputs."""


class ConfigurationError(ACCT445Error):
    """Raised when user configuration or parameters are invalid."""
