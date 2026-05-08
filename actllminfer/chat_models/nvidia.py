"""NVIDIA serverless inference chat model.

NVIDIA's NIM (build.nvidia.com) exposes a free, OpenAI-compatible
``/v1/chat/completions`` endpoint at ``https://integrate.api.nvidia.com/v1``.
A free-tier ``NVIDIA_API_KEY`` (``nvapi-...``) can be obtained from
https://build.nvidia.com/ — the same key fans out to dozens of hosted models
(Llama, Mixtral, Nemotron, Qwen, DeepSeek, Phi, Gemma, ...).

Example::

    from actllminfer import ChatNVIDIA

    llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0.2)
    print(llm.invoke("Say hi in one word.").content)

The class also accepts the older ``NGC_API_KEY`` env var as a fallback.
"""

from __future__ import annotations

import os
from typing import Any

from ..utils import SecretStr
from ._openai_compat import OpenAICompatibleChat


_LEGACY_KEY_ENV = "NGC_API_KEY"


class ChatNVIDIA(OpenAICompatibleChat):
    PROVIDER_NAME = "nvidia"
    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    API_KEY_ENV = "NVIDIA_API_KEY"
    BASE_URL_ENV = "NVIDIA_BASE_URL"
    DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | SecretStr | None = None,
        **kwargs: Any,
    ) -> None:
        if api_key is None and not os.environ.get(self.API_KEY_ENV):
            legacy = os.environ.get(_LEGACY_KEY_ENV)
            if legacy:
                api_key = legacy
        super().__init__(model=model, api_key=api_key, **kwargs)


__all__ = ["ChatNVIDIA"]
