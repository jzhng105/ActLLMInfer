"""Exception hierarchy for actllminfer."""

from __future__ import annotations


class LLMError(Exception):
    """Base class for all errors raised by actllminfer."""


class ConfigurationError(LLMError):
    """Raised when a chat model is misconfigured (e.g., missing API key)."""


class AuthenticationError(LLMError):
    """Raised when a provider rejects credentials (HTTP 401/403)."""


class RateLimitError(LLMError):
    """Raised when a provider returns HTTP 429."""


class APIError(LLMError):
    """Raised for non-retriable HTTP errors from a provider."""

    def __init__(self, message: str, status_code: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class APIConnectionError(LLMError):
    """Raised when the HTTP transport itself fails (DNS, timeout, reset)."""


class OutputParserError(LLMError):
    """Raised when an output parser cannot parse model output."""
