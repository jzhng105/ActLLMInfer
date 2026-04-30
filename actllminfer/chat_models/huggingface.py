"""Hugging Face chat model.

Uses the Hugging Face Inference Router, which exposes an OpenAI-compatible
``/v1/chat/completions`` endpoint that automatically dispatches to whichever
inference provider is available for a given ``user/model`` id::

    https://router.huggingface.co/v1/chat/completions

Pointing ``base_url`` at any other ``/v1``-rooted server (a TGI deployment,
a dedicated HF Inference Endpoint, vLLM, etc.) keeps the same code path:

    ChatHuggingFace(model="...", base_url="https://my-endpoint.example/v1")
"""

from __future__ import annotations

import os
from typing import Any

from ..utils import SecretStr
from ._openai_compat import OpenAICompatibleChat


_LEGACY_TOKEN_ENV = "HUGGINGFACEHUB_API_TOKEN"


class ChatHuggingFace(OpenAICompatibleChat):
    PROVIDER_NAME = "huggingface"
    DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
    API_KEY_ENV = "HF_TOKEN"
    BASE_URL_ENV = "HF_BASE_URL"
    DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | SecretStr | None = None,
        **kwargs: Any,
    ) -> None:
        # Honor the older HF env var name if HF_TOKEN is not set.
        if api_key is None and not os.environ.get(self.API_KEY_ENV):
            legacy = os.environ.get(_LEGACY_TOKEN_ENV)
            if legacy:
                api_key = legacy
        super().__init__(model=model, api_key=api_key, **kwargs)


__all__ = ["ChatHuggingFace"]