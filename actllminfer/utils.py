"""Small shared helpers: env lookup, retry, secret string."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Iterable, TypeVar

from .exceptions import APIConnectionError, ConfigurationError, RateLimitError

T = TypeVar("T")


def get_from_env_or_arg(value: str | None, env_var: str, name: str) -> str:
    """Resolve a credential or endpoint from an explicit arg or environment variable."""
    if value:
        return value
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value
    raise ConfigurationError(
        f"{name} was not provided and environment variable {env_var} is not set."
    )


def get_optional_env(value: str | None, env_var: str) -> str | None:
    if value:
        return value
    return os.environ.get(env_var) or None


class SecretStr:
    """Wrap a secret so it does not leak through repr/str."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('**********')"

    def __str__(self) -> str:
        return "**********"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SecretStr) and other._value == self._value

    def __hash__(self) -> int:
        return hash(self._value)


def retry_call(
    fn: Callable[[], T],
    *,
    max_retries: int = 2,
    initial_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on: Iterable[type[BaseException]] = (RateLimitError, APIConnectionError),
) -> T:
    """Call ``fn``; retry on the listed exception types with exponential backoff."""
    retry_tuple = tuple(retry_on)
    attempt = 0
    backoff = initial_backoff
    while True:
        try:
            return fn()
        except retry_tuple as exc:  # type: ignore[misc]
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_for = backoff
            backoff *= backoff_factor
            time.sleep(sleep_for)
            _ = exc  # silence unused


def filter_none(d: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``d`` with ``None`` values removed."""
    return {k: v for k, v in d.items() if v is not None}
