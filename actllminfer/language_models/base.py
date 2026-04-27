"""``BaseChatModel`` — provider-agnostic interface for chat models.

Subclasses implement ``_generate`` (required) and ``_stream`` (optional).
This base handles message coercion, callbacks, retry, and the
``invoke`` / ``batch`` / ``stream`` Runnable surface.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator, Sequence

from ..callbacks import BaseCallbackHandler, CallbackManager
from ..exceptions import APIConnectionError, RateLimitError
from ..messages import AIMessage, AIMessageChunk, BaseMessage, MessageLike, coerce_messages
from ..outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from ..runnables import Runnable
from ..utils import retry_call


class BaseChatModel(Runnable):
    """Common base for all chat model providers."""

    #: e.g. ``moonshot-v1-8k``, ``glm-4-plus``. Set by the subclass constructor.
    model: str
    #: Per-instance callbacks. Plus the per-call ``callbacks=...`` kwarg.
    callbacks: list[BaseCallbackHandler]
    #: Number of additional attempts on transient errors.
    max_retries: int

    def __init__(
        self,
        *,
        model: str,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        max_retries: int = 2,
        request_timeout: float | tuple[float, float] = 60.0,
    ) -> None:
        self.model = model
        self.callbacks = list(callbacks or [])
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult: ...

    def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Default: emit the non-streaming result as a single chunk."""
        result = self._generate(messages, **kwargs)
        msg = result.message
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(
                content=msg.content,
                additional_kwargs=dict(msg.additional_kwargs),
                tool_calls=list(msg.tool_calls),
            ),
            generation_info=result.llm_output,
        )
        yield chunk

    # ------------------------------------------------------------------ #
    # Public Runnable surface
    # ------------------------------------------------------------------ #

    def invoke(
        self,
        input: str | MessageLike | Sequence[MessageLike],
        *,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        result = self.generate(input, callbacks=callbacks, **kwargs)
        return result.message

    def generate(
        self,
        input: str | MessageLike | Sequence[MessageLike],
        *,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        messages = coerce_messages(input)
        cb = CallbackManager([*self.callbacks, *(callbacks or [])])
        cb.on_llm_start({"model": self.model, "type": type(self).__name__}, messages, **kwargs)
        try:
            result = retry_call(
                lambda: self._generate(messages, **kwargs),
                max_retries=self.max_retries,
                retry_on=(RateLimitError, APIConnectionError),
            )
        except BaseException as exc:
            cb.on_llm_error(exc)
            raise
        cb.on_llm_end(result)
        return result

    def stream(
        self,
        input: str | MessageLike | Sequence[MessageLike],
        *,
        callbacks: Sequence[BaseCallbackHandler] | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        messages = coerce_messages(input)
        cb = CallbackManager([*self.callbacks, *(callbacks or [])])
        cb.on_llm_start({"model": self.model, "type": type(self).__name__}, messages, **kwargs)
        accumulated: AIMessageChunk | None = None
        try:
            for chunk in self._stream(messages, **kwargs):
                accumulated = chunk.message if accumulated is None else accumulated + chunk.message
                cb.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk
        except BaseException as exc:
            cb.on_llm_error(exc)
            raise
        if accumulated is not None:
            final = ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=accumulated.content,
                            additional_kwargs=accumulated.additional_kwargs,
                            tool_calls=accumulated.tool_calls,
                        )
                    )
                ]
            )
            cb.on_llm_end(final)

    def batch(self, inputs, **kwargs: Any) -> list[AIMessage]:  # type: ignore[override]
        return [self.invoke(i, **kwargs) for i in inputs]

    # ------------------------------------------------------------------ #
    # Convenience
    # ------------------------------------------------------------------ #

    def bind(self, **kwargs: Any) -> "BaseChatModel":
        """Return a thin wrapper that injects default kwargs into every call."""
        return _BoundChatModel(self, kwargs)

    def with_tools(self, tools: list[dict] | list[Any], **kwargs: Any) -> "BaseChatModel":
        """Bind a tool spec list (OpenAI-style ``tools=[...]``) to every call."""
        return self.bind(tools=tools, **kwargs)


class _BoundChatModel(BaseChatModel):
    """Wraps a chat model with default kwargs that are merged into every call."""

    def __init__(self, inner: BaseChatModel, defaults: dict[str, Any]) -> None:
        super().__init__(
            model=inner.model,
            callbacks=inner.callbacks,
            max_retries=inner.max_retries,
            request_timeout=inner.request_timeout,
        )
        self._inner = inner
        self._defaults = defaults

    def _merge(self, kwargs: dict) -> dict:
        merged = dict(self._defaults)
        merged.update(kwargs)
        return merged

    def _generate(self, messages: list[BaseMessage], **kwargs: Any) -> ChatResult:
        return self._inner._generate(messages, **self._merge(kwargs))

    def _stream(self, messages: list[BaseMessage], **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        return self._inner._stream(messages, **self._merge(kwargs))


__all__ = ["BaseChatModel"]
