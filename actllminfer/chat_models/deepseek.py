"""DeepSeek chat model.

DeepSeek serves an OpenAI-compatible API at https://api.deepseek.com.
The reasoner family also returns a ``reasoning_content`` field which the
shared backend already exposes via ``additional_kwargs["reasoning_content"]``.
"""

from __future__ import annotations

from ._openai_compat import OpenAICompatibleChat


class ChatDeepSeek(OpenAICompatibleChat):
    PROVIDER_NAME = "deepseek"
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    API_KEY_ENV = "DEEPSEEK_API_KEY"
    BASE_URL_ENV = "DEEPSEEK_BASE_URL"
    DEFAULT_MODEL = "deepseek-chat"


__all__ = ["ChatDeepSeek"]
