"""Lightweight callback system, modeled on LangChain Core's BaseCallbackHandler."""

from __future__ import annotations

from typing import Any, Sequence

from .messages import BaseMessage
from .outputs import ChatGenerationChunk, ChatResult


class BaseCallbackHandler:
    """Override the hooks you care about. All hooks are no-ops by default."""

    def on_llm_start(self, serialized: dict, messages: Sequence[BaseMessage], **kwargs: Any) -> None:  # noqa: D401
        ...

    def on_llm_new_token(self, token: str, *, chunk: ChatGenerationChunk | None = None, **kwargs: Any) -> None:
        ...

    def on_llm_end(self, response: ChatResult, **kwargs: Any) -> None:
        ...

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        ...


class CallbackManager:
    """Fan-out helper that swallows handler exceptions so they cannot break inference."""

    def __init__(self, handlers: Sequence[BaseCallbackHandler] | None = None) -> None:
        self.handlers: list[BaseCallbackHandler] = list(handlers or [])

    def add(self, handler: BaseCallbackHandler) -> None:
        self.handlers.append(handler)

    def _safe(self, name: str, *args: Any, **kwargs: Any) -> None:
        for h in self.handlers:
            try:
                getattr(h, name)(*args, **kwargs)
            except Exception:  # callbacks must never affect the response
                pass

    def on_llm_start(self, serialized: dict, messages: Sequence[BaseMessage], **kwargs: Any) -> None:
        self._safe("on_llm_start", serialized, messages, **kwargs)

    def on_llm_new_token(self, token: str, *, chunk: ChatGenerationChunk | None = None, **kwargs: Any) -> None:
        self._safe("on_llm_new_token", token, chunk=chunk, **kwargs)

    def on_llm_end(self, response: ChatResult, **kwargs: Any) -> None:
        self._safe("on_llm_end", response, **kwargs)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self._safe("on_llm_error", error, **kwargs)


class StdOutCallbackHandler(BaseCallbackHandler):
    """Print streamed tokens to stdout. Handy for demos and CLIs."""

    def __init__(self, end: str = "", flush: bool = True) -> None:
        self.end = end
        self.flush = flush

    def on_llm_new_token(self, token: str, *, chunk: ChatGenerationChunk | None = None, **kwargs: Any) -> None:
        print(token, end=self.end, flush=self.flush)


__all__ = ["BaseCallbackHandler", "CallbackManager", "StdOutCallbackHandler"]
