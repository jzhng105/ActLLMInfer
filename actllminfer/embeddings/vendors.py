from __future__ import annotations

from .openai_compat import OpenAICompatibleEmbeddings


class OpenAIEmbeddings(OpenAICompatibleEmbeddings):
    PROVIDER_NAME = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    API_KEY_ENV = "OPENAI_API_KEY"
    BASE_URL_ENV = "OPENAI_BASE_URL"
    DEFAULT_MODEL = "text-embedding-3-small"


class DeepSeekEmbeddings(OpenAICompatibleEmbeddings):
    PROVIDER_NAME = "deepseek"
    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    API_KEY_ENV = "DEEPSEEK_API_KEY"
    BASE_URL_ENV = "DEEPSEEK_BASE_URL"
    DEFAULT_MODEL = "deepseek-embedding"


class ZhipuAIEmbeddings(OpenAICompatibleEmbeddings):
    PROVIDER_NAME = "zhipuai"
    DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
    API_KEY_ENV = "ZHIPUAI_API_KEY"
    BASE_URL_ENV = "ZHIPUAI_BASE_URL"
    DEFAULT_MODEL = "embedding-3"
