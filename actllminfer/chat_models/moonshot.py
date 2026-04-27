"""Moonshot (Kimi) chat model.

Moonshot's API is OpenAI-compatible: https://platform.moonshot.cn/docs
"""

from __future__ import annotations

from ._openai_compat import OpenAICompatibleChat


class ChatMoonshot(OpenAICompatibleChat):
    PROVIDER_NAME = "moonshot"
    DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
    API_KEY_ENV = "MOONSHOT_API_KEY"
    BASE_URL_ENV = "MOONSHOT_BASE_URL"
    DEFAULT_MODEL = "moonshot-v1-8k"


# "Kimi" is the consumer-facing brand name; expose an alias.
ChatKimi = ChatMoonshot


__all__ = ["ChatMoonshot", "ChatKimi"]
