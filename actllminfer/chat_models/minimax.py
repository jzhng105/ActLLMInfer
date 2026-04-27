"""MiniMax chat model.

Uses MiniMax's OpenAI-style v2 chat completion endpoint:
    https://api.minimax.chat/v1/text/chatcompletion_v2
"""

from __future__ import annotations

from ._openai_compat import OpenAICompatibleChat


class ChatMiniMax(OpenAICompatibleChat):
    PROVIDER_NAME = "minimax"
    DEFAULT_BASE_URL = "https://api.minimax.chat/v1"
    API_KEY_ENV = "MINIMAX_API_KEY"
    BASE_URL_ENV = "MINIMAX_BASE_URL"
    DEFAULT_MODEL = "abab6.5s-chat"

    def _chat_endpoint(self) -> str:
        return f"{self.base_url}/text/chatcompletion_v2"


__all__ = ["ChatMiniMax"]
