"""LiteLLM-style unified entry point for chat completions and embeddings.

Single ``completion()`` and ``embedding()`` function across every provider
exposed by :mod:`actllminfer`. The model is selected by a single string and
the response shape mirrors OpenAI's ``/v1/chat/completions`` so callers can
swap providers without changing downstream parsing.

Examples::

    from actllminfer import completion, embedding

    resp = completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
        temperature=0.2,
    )
    print(resp["choices"][0]["message"]["content"])
    print(resp.choices[0].message.content)  # attribute access works too

    for chunk in completion(model="kimi/moonshot-v1-8k", messages=msgs, stream=True):
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="")

    vec = embedding(model="openai/text-embedding-3-small", input=["hi"])
    print(vec.data[0].embedding)

The ``Router`` class adds primary-with-fallbacks dispatch::

    router = Router([
        "openai/gpt-4o-mini",
        "kimi/moonshot-v1-8k",
        "deepseek/deepseek-chat",
    ])
    resp = router.completion(messages=msgs)
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from threading import RLock
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .embeddings.base import BaseEmbeddings
from .exceptions import ConfigurationError, LLMError
from .factory import (
    _EMBEDDING_PROVIDER_REGISTRY,
    _PROVIDER_REGISTRY,
    _infer_provider,
)
from .language_models.base import BaseChatModel
from .messages import BaseMessage, coerce_messages
from .outputs import ChatGenerationChunk, ChatResult


# ---------------------------------------------------------------------------
# Response containers (dict subclasses with attribute access)
# ---------------------------------------------------------------------------


class _AttrDict(dict):
    """Dict that also exposes its keys as attributes."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class Message(_AttrDict):
    """``choice.message`` — assistant message in a completion response."""


class Delta(_AttrDict):
    """``choice.delta`` — incremental assistant message in a streaming chunk."""


class Choice(_AttrDict):
    """One entry of ``response.choices`` for a non-streaming completion."""


class StreamingChoice(_AttrDict):
    """One entry of ``chunk.choices`` for a streaming completion."""


class Usage(_AttrDict):
    """Token-usage block returned by the provider."""


class ModelResponse(_AttrDict):
    """Full OpenAI-shaped chat-completion response."""

    @property
    def text(self) -> str:
        choices = self.get("choices") or []
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content") or ""


class StreamingModelResponse(_AttrDict):
    """One chunk of a streaming response (``chat.completion.chunk``)."""


class EmbeddingObject(_AttrDict):
    """One row of an embeddings response."""


class EmbeddingResponse(_AttrDict):
    """Full OpenAI-shaped embeddings response."""


# ---------------------------------------------------------------------------
# Model-string parsing
# ---------------------------------------------------------------------------
_NAMESPACED_MODEL_PROVIDERS: frozenset[str] = frozenset({"nvidia", "nim"})

def _split_provider_and_model(spec: str) -> tuple[str | None, str]:
    """Pull ``provider`` off the front of ``provider/model`` or ``provider:model``.

    Both separators are supported. The ``/`` form is the LiteLLM convention,
    but several model ids themselves contain ``/`` (``nvidia/llama-...``,
    ``meta-llama/Llama-3.3-70B-Instruct``), so the head is only treated as a
    provider when it appears in :data:`_PROVIDER_REGISTRY`.
    """
    for sep in (":", "/"):
        if sep in spec:
            head, tail = spec.split(sep, 1)
            head_lower = head.lower()
            if head_lower in _PROVIDER_REGISTRY or head_lower in _EMBEDDING_PROVIDER_REGISTRY:
                if (
                    sep == "/"
                    and head_lower in _NAMESPACED_MODEL_PROVIDERS
                    and "/" not in tail
                ):
                    return None, spec
            return head_lower, tail  
    return None, spec


def _resolve_provider(model: str, provider: str | None, registry: Mapping[str, Any]) -> tuple[str, str]:
    spec = model
    if provider is None:
        provider, spec = _split_provider_and_model(model)
    if provider is None:
        provider = _infer_provider(spec)
    if provider is None:
        raise ConfigurationError(
            f"Could not infer provider for model {model!r}. "
            f"Pass provider= explicitly. Known providers: {sorted(set(registry))}"
        )
    if provider.lower() not in registry:
        raise ConfigurationError(
            f"Unknown provider {provider!r}. Known providers: {sorted(set(registry))}"
        )
    return provider.lower(), spec


# ---------------------------------------------------------------------------
# Client cache
# ---------------------------------------------------------------------------


_CLIENT_LOCK = RLock()
_CHAT_CLIENT_CACHE: dict[tuple, BaseChatModel] = {}
_EMBEDDING_CLIENT_CACHE: dict[tuple, BaseEmbeddings] = {}


_PER_CALL_KWARGS: frozenset[str] = frozenset({
    # OpenAI-compatible payload knobs that vary per call.
    "temperature", "top_p", "max_tokens", "stop", "stop_sequences",
    "tools", "tool_choice", "response_format", "seed", "user",
    "presence_penalty", "frequency_penalty", "logit_bias", "n",
    "stream", "callbacks", "logprobs", "top_logprobs", "metadata",
})


def _split_init_and_call_kwargs(kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Partition kwargs into constructor-time vs per-call arguments.

    Anything in :data:`_PER_CALL_KWARGS` is forwarded to ``_generate`` /
    ``_stream``; everything else (``api_key``, ``base_url``, ``organization``,
    ``request_timeout``, etc.) is used to build the cached client.
    """
    init_kwargs: dict[str, Any] = {}
    call_kwargs: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _PER_CALL_KWARGS:
            call_kwargs[k] = v
        else:
            init_kwargs[k] = v
    return init_kwargs, call_kwargs


def _cache_key(provider: str, model: str, init_kwargs: dict[str, Any]) -> tuple:
    items = []
    for k, v in sorted(init_kwargs.items()):
        try:
            hash(v)
            items.append((k, v))
        except TypeError:
            items.append((k, repr(v)))
    return (provider, model, tuple(items))


def _get_chat_client(provider: str, model: str, init_kwargs: dict[str, Any]) -> BaseChatModel:
    key = _cache_key(provider, model, init_kwargs)
    with _CLIENT_LOCK:
        client = _CHAT_CLIENT_CACHE.get(key)
        if client is None:
            cls = _PROVIDER_REGISTRY[provider]
            client = cls(model=model, **init_kwargs)
            _CHAT_CLIENT_CACHE[key] = client
        return client


def _get_embedding_client(provider: str, model: str, init_kwargs: dict[str, Any]) -> BaseEmbeddings:
    key = _cache_key(provider, model, init_kwargs)
    with _CLIENT_LOCK:
        client = _EMBEDDING_CLIENT_CACHE.get(key)
        if client is None:
            cls = _EMBEDDING_PROVIDER_REGISTRY[provider]
            client = cls(model=model, **init_kwargs)
            _EMBEDDING_CLIENT_CACHE[key] = client
        return client


def reset_client_cache() -> None:
    """Drop every cached provider client. Intended for tests."""
    with _CLIENT_LOCK:
        _CHAT_CLIENT_CACHE.clear()
        _EMBEDDING_CLIENT_CACHE.clear()


# ---------------------------------------------------------------------------
# Conversion: internal ChatResult -> OpenAI-shaped ModelResponse
# ---------------------------------------------------------------------------


def _tool_calls_to_openai(message_tool_calls: Sequence[Any]) -> list[dict[str, Any]] | None:
    if not message_tool_calls:
        return None
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.name,
                "arguments": json.dumps(tc.args, ensure_ascii=False),
            },
        }
        for tc in message_tool_calls
    ]


def _chat_result_to_model_response(
    result: ChatResult, *, requested_model: str, provider: str
) -> ModelResponse:
    llm_output = result.llm_output or {}
    choices: list[Choice] = []
    for idx, gen in enumerate(result.generations):
        msg = gen.message
        body: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content if isinstance(msg.content, str) else msg.content,
        }
        tool_calls = _tool_calls_to_openai(msg.tool_calls)
        if tool_calls is not None:
            body["tool_calls"] = tool_calls
        if msg.additional_kwargs:
            for k, v in msg.additional_kwargs.items():
                body.setdefault(k, v)
        info = gen.generation_info or {}
        choices.append(
            Choice(
                index=info.get("index", idx),
                message=Message(body),
                finish_reason=info.get("finish_reason") or info.get("stop_reason"),
            )
        )
    response = ModelResponse(
        id=llm_output.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        object="chat.completion",
        created=int(time.time()),
        model=llm_output.get("model") or requested_model,
        provider=llm_output.get("provider") or provider,
        choices=choices,
        usage=Usage(llm_output.get("usage") or {}),
    )
    if llm_output.get("system_fingerprint"):
        response["system_fingerprint"] = llm_output["system_fingerprint"]
    return response


def _chunk_to_streaming_response(
    chunk: ChatGenerationChunk, *, response_id: str, requested_model: str, provider: str
) -> StreamingModelResponse:
    msg = chunk.message
    delta: dict[str, Any] = {}
    if msg.content:
        delta["content"] = msg.content if isinstance(msg.content, str) else msg.content
    tool_calls = _tool_calls_to_openai(msg.tool_calls)
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    if msg.additional_kwargs:
        for k, v in msg.additional_kwargs.items():
            delta.setdefault(k, v)
    info = chunk.generation_info or {}
    return StreamingModelResponse(
        id=response_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=requested_model,
        provider=provider,
        choices=[
            StreamingChoice(
                index=0,
                delta=Delta(delta),
                finish_reason=info.get("finish_reason") or info.get("stop_reason"),
            )
        ],
    )


# ---------------------------------------------------------------------------
# Public API: completion
# ---------------------------------------------------------------------------


def _coerce_input_messages(
    messages: str | Sequence[Mapping[str, Any]] | Sequence[BaseMessage] | BaseMessage,
) -> list[BaseMessage]:
    return coerce_messages(messages)


def completion(
    model: str,
    messages: str | Sequence[Mapping[str, Any]] | Sequence[BaseMessage] | BaseMessage,
    *,
    provider: str | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> ModelResponse | Iterator[StreamingModelResponse]:
    """Unified chat-completion entry point across every supported provider.

    The ``model`` argument accepts ``provider/model`` (LiteLLM style),
    ``provider:model``, or a bare model id whose prefix can be inferred
    (``glm-4-plus``, ``deepseek-chat``, ``claude-sonnet-4-6``, ...).

    Per-call generation knobs (``temperature``, ``max_tokens``, ``tools``,
    ``response_format``, ``seed``, ``stop``, ...) are forwarded to the
    provider. Everything else (``api_key``, ``base_url``, ``organization``,
    ``request_timeout``, ``default_headers``, ``extra_body``, ...) is used
    to construct (and cache) the underlying client.

    Returns an OpenAI-shaped :class:`ModelResponse`, or an iterator of
    :class:`StreamingModelResponse` chunks when ``stream=True``.
    """
    provider_name, model_id = _resolve_provider(model, provider, _PROVIDER_REGISTRY)
    init_kwargs, call_kwargs = _split_init_and_call_kwargs(kwargs)
    client = _get_chat_client(provider_name, model_id, init_kwargs)
    msgs = _coerce_input_messages(messages)

    if stream:
        response_id = f"chatcmpl-{uuid.uuid4().hex}"

        def _iter() -> Iterator[StreamingModelResponse]:
            for chunk in client.stream(msgs, **call_kwargs):
                yield _chunk_to_streaming_response(
                    chunk,
                    response_id=response_id,
                    requested_model=model_id,
                    provider=provider_name,
                )

        return _iter()

    result = client.generate(msgs, **call_kwargs)
    return _chat_result_to_model_response(
        result, requested_model=model_id, provider=provider_name
    )


async def acompletion(
    model: str,
    messages: str | Sequence[Mapping[str, Any]] | Sequence[BaseMessage] | BaseMessage,
    *,
    provider: str | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> ModelResponse | "AsyncStreamIterator":
    """Async wrapper around :func:`completion`.

    The synchronous provider call runs on a background thread, so callers
    inside an event loop don't block it. Streaming returns an async iterator
    of chunks.
    """
    if stream:
        result = await asyncio.to_thread(
            completion, model, messages, provider=provider, stream=True, **kwargs
        )
        return AsyncStreamIterator(result)  # type: ignore[arg-type]
    return await asyncio.to_thread(
        completion, model, messages, provider=provider, stream=False, **kwargs
    )


class AsyncStreamIterator:
    """Wrap a sync chunk iterator so callers can ``async for`` over it."""

    def __init__(self, iterator: Iterator[StreamingModelResponse]) -> None:
        self._iterator = iterator

    def __aiter__(self) -> "AsyncStreamIterator":
        return self

    async def __anext__(self) -> StreamingModelResponse:
        sentinel = object()
        chunk = await asyncio.to_thread(next, self._iterator, sentinel)
        if chunk is sentinel:
            raise StopAsyncIteration
        return chunk  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API: embedding
# ---------------------------------------------------------------------------


def embedding(
    model: str,
    input: str | Sequence[str],  # noqa: A002 - mirror OpenAI's parameter name
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> EmbeddingResponse:
    """Unified embeddings entry point.

    Mirrors OpenAI's ``/v1/embeddings`` response shape::

        {"object": "list", "data": [{"object": "embedding", "embedding": [...], "index": 0}, ...],
         "model": "...", "usage": {...}}
    """
    provider_name, model_id = _resolve_provider(model, provider, _EMBEDDING_PROVIDER_REGISTRY)
    init_kwargs, call_kwargs = _split_init_and_call_kwargs(kwargs)
    client = _get_embedding_client(provider_name, model_id, init_kwargs)

    texts = [input] if isinstance(input, str) else list(input)
    vectors = client.embed_documents(texts, **call_kwargs)
    rows = [
        EmbeddingObject(object="embedding", embedding=vec, index=idx)
        for idx, vec in enumerate(vectors)
    ]
    return EmbeddingResponse(
        object="list",
        data=rows,
        model=model_id,
        provider=provider_name,
        usage=Usage(),
    )


async def aembedding(
    model: str,
    input: str | Sequence[str],  # noqa: A002
    *,
    provider: str | None = None,
    **kwargs: Any,
) -> EmbeddingResponse:
    """Async wrapper around :func:`embedding`."""
    return await asyncio.to_thread(
        embedding, model, input, provider=provider, **kwargs
    )


# ---------------------------------------------------------------------------
# Router: primary-with-fallbacks dispatch
# ---------------------------------------------------------------------------


class Router:
    """Try a list of models in order, falling back on transient errors.

    Useful when a primary provider is rate-limited or unreachable::

        router = Router([
            "openai/gpt-4o-mini",
            "kimi/moonshot-v1-8k",
            "deepseek/deepseek-chat",
        ])
        resp = router.completion(messages=[{"role": "user", "content": "Hi"}])
    """

    def __init__(
        self,
        models: Sequence[str | Mapping[str, Any]],
        *,
        default_params: Mapping[str, Any] | None = None,
    ) -> None:
        if not models:
            raise ConfigurationError("Router requires at least one model")
        self._entries: list[dict[str, Any]] = []
        for entry in models:
            if isinstance(entry, str):
                self._entries.append({"model": entry})
            else:
                if "model" not in entry:
                    raise ConfigurationError("Router model entry must include a 'model' key")
                self._entries.append(dict(entry))
        self._default_params = dict(default_params or {})

    def _merged(self, entry: dict[str, Any], call_kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        params = dict(self._default_params)
        params.update({k: v for k, v in entry.items() if k != "model"})
        params.update(call_kwargs)
        return entry["model"], params

    def completion(
        self,
        messages: str | Sequence[Mapping[str, Any]] | Sequence[BaseMessage] | BaseMessage,
        **kwargs: Any,
    ) -> ModelResponse | Iterator[StreamingModelResponse]:
        last_exc: BaseException | None = None
        for entry in self._entries:
            model, params = self._merged(entry, kwargs)
            try:
                return completion(model=model, messages=messages, **params)
            except LLMError as exc:
                last_exc = exc
                continue
        assert last_exc is not None
        raise last_exc

    def embedding(
        self,
        input: str | Sequence[str],  # noqa: A002
        **kwargs: Any,
    ) -> EmbeddingResponse:
        last_exc: BaseException | None = None
        for entry in self._entries:
            model, params = self._merged(entry, kwargs)
            try:
                return embedding(model=model, input=input, **params)
            except LLMError as exc:
                last_exc = exc
                continue
        assert last_exc is not None
        raise last_exc


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def supported_chat_providers() -> list[str]:
    """Return the list of provider names accepted by :func:`completion`."""
    return sorted(set(_PROVIDER_REGISTRY))


def supported_embedding_providers() -> list[str]:
    """Return the list of provider names accepted by :func:`embedding`."""
    return sorted(set(_EMBEDDING_PROVIDER_REGISTRY))


__all__ = [
    "completion",
    "acompletion",
    "embedding",
    "aembedding",
    "Router",
    "ModelResponse",
    "StreamingModelResponse",
    "Choice",
    "StreamingChoice",
    "Message",
    "Delta",
    "Usage",
    "EmbeddingResponse",
    "EmbeddingObject",
    "AsyncStreamIterator",
    "supported_chat_providers",
    "supported_embedding_providers",
    "reset_client_cache",
]
