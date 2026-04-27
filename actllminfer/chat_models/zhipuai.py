"""ZhipuAI (GLM) chat model.

ZhipuAI's v4 API is OpenAI-compatible:
    https://open.bigmodel.cn/api/paas/v4/chat/completions
"""

from __future__ import annotations

from ._openai_compat import OpenAICompatibleChat


class ChatZhipuAI(OpenAICompatibleChat):
    PROVIDER_NAME = "zhipuai"
    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    API_KEY_ENV = "ZHIPUAI_API_KEY"
    BASE_URL_ENV = "ZHIPUAI_BASE_URL"
    DEFAULT_MODEL = "glm-4-plus"


# Common alias since most users refer to the family by model name.
ChatGLM = ChatZhipuAI


__all__ = ["ChatZhipuAI", "ChatGLM"]
