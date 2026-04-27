"""OpenAI chat model."""

from __future__ import annotations

from ._openai_compat import OpenAICompatibleChat


class ChatOpenAI(OpenAICompatibleChat):
    PROVIDER_NAME = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    API_KEY_ENV = "OPENAI_API_KEY"
    BASE_URL_ENV = "OPENAI_BASE_URL"
    DEFAULT_MODEL = "gpt-4o-mini"


__all__ = ["ChatOpenAI"]
