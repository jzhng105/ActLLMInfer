"""``init_chat_model`` — string-spec entry point for any supported provider.

Examples::

    init_chat_model("kimi:moonshot-v1-8k")
    init_chat_model("glm-4-plus", provider="zhipuai")
    init_chat_model("deepseek-reasoner", provider="deepseek", temperature=0)
    init_chat_model("openai:gpt-4o-mini")
    init_chat_model("anthropic:claude-sonnet-4-6")
"""

from __future__ import annotations

from typing import Any

from .chat_models import (
    ChatAnthropic,
    ChatDeepSeek,
    ChatHuggingFace,
    ChatMiniMax,
    ChatMoonshot,
    ChatOpenAI,
    ChatZhipuAI,
)
from .exceptions import ConfigurationError
from .language_models.base import BaseChatModel


_PROVIDER_REGISTRY: dict[str, type[BaseChatModel]] = {
    "openai": ChatOpenAI,
    "moonshot": ChatMoonshot,
    "kimi": ChatMoonshot,
    "deepseek": ChatDeepSeek,
    "zhipuai": ChatZhipuAI,
    "zhipu": ChatZhipuAI,
    "glm": ChatZhipuAI,
    "minimax": ChatMiniMax,
    "anthropic": ChatAnthropic,
    "claude": ChatAnthropic,
    "huggingface": ChatHuggingFace,
    "hf": ChatHuggingFace,
}


_MODEL_PREFIX_HINTS: list[tuple[str, str]] = [
    ("moonshot", "moonshot"),
    ("kimi", "moonshot"),
    ("deepseek", "deepseek"),
    ("glm", "zhipuai"),
    ("chatglm", "zhipuai"),
    ("abab", "minimax"),
    ("minimax", "minimax"),
    ("claude", "anthropic"),
    ("gpt", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
]


def _infer_provider(model: str) -> str | None:
    lowered = model.lower()
    for prefix, provider in _MODEL_PREFIX_HINTS:
        if lowered.startswith(prefix):
            return provider
    return None


def init_chat_model(
    model: str,
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a chat model from a string spec.

    Accepted forms:
      * ``"openai:gpt-4o-mini"`` — explicit ``provider:model`` prefix
      * ``"kimi:moonshot-v1-8k"``
      * ``"moonshot-v1-8k"`` with ``provider="moonshot"``
      * ``"glm-4-plus"`` (provider inferred from the ``glm-`` prefix)
    """
    spec = model
    if ":" in model and provider is None:
        head, tail = model.split(":", 1)
        if head.lower() in _PROVIDER_REGISTRY:
            provider = head
            spec = tail
    if provider is None:
        provider = _infer_provider(spec)
    if provider is None:
        raise ConfigurationError(
            f"Could not infer provider for model {model!r}. "
            f"Pass provider= explicitly. Known providers: {sorted(set(_PROVIDER_REGISTRY))}"
        )
    cls = _PROVIDER_REGISTRY.get(provider.lower())
    if cls is None:
        raise ConfigurationError(
            f"Unknown provider {provider!r}. Known providers: {sorted(set(_PROVIDER_REGISTRY))}"
        )
    return cls(model=spec, **kwargs)


__all__ = ["init_chat_model"]
