"""Output parsers convert a chat model response into a downstream Python value."""

from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any, Generic, Iterator, TypeVar

from .exceptions import OutputParserError
from .messages import AIMessage, BaseMessage
from .outputs import ChatGenerationChunk, ChatResult
from .runnables import Runnable

T = TypeVar("T")


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, BaseMessage):
        c = value.content
        if isinstance(c, str):
            return c
        return "".join(p if isinstance(p, str) else p.get("text", "") for p in c)
    if isinstance(value, ChatResult):
        return value.text
    if isinstance(value, ChatGenerationChunk):
        return value.text
    if isinstance(value, dict) and "content" in value:
        return _to_text(value["content"])
    return str(value)


class BaseOutputParser(Runnable[Any, T], Generic[T]):
    @abstractmethod
    def parse(self, text: str) -> T: ...

    def invoke(self, input: Any, **kwargs: Any) -> T:
        return self.parse(_to_text(input))

    def stream(self, input: Any, **kwargs: Any) -> Iterator[T]:
        # Most parsers are not naturally streamable; fold the stream first.
        if hasattr(input, "__iter__") and not isinstance(input, (str, bytes, BaseMessage)):
            chunks = list(input)
            text = "".join(_to_text(c) for c in chunks)
            yield self.parse(text)
        else:
            yield self.invoke(input, **kwargs)


class StrOutputParser(BaseOutputParser[str]):
    """Return the assistant text unchanged."""

    def parse(self, text: str) -> str:
        return text

    def stream(self, input: Any, **kwargs: Any) -> Iterator[str]:
        if hasattr(input, "__iter__") and not isinstance(input, (str, bytes, BaseMessage)):
            for chunk in input:
                yield _to_text(chunk)
        else:
            yield self.invoke(input, **kwargs)


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class JsonOutputParser(BaseOutputParser[Any]):
    """Parse a JSON object from the model output, tolerating ``` fences."""

    def parse(self, text: str) -> Any:
        candidate = self._extract_json_blob(text)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise OutputParserError(
                f"Could not parse JSON from output: {exc.msg}\n--- raw ---\n{text}"
            ) from exc

    @staticmethod
    def _extract_json_blob(text: str) -> str:
        fenced = _JSON_FENCE.search(text)
        if fenced:
            return fenced.group(1).strip()
        stripped = text.strip()
        # Find first { or [ and the matching end so we tolerate prose around the object.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = stripped.find(opener)
            if start != -1:
                end = stripped.rfind(closer)
                if end > start:
                    return stripped[start : end + 1]
        return stripped


class CommaSeparatedListOutputParser(BaseOutputParser[list[str]]):
    """Split a comma-separated string from the model into a list of trimmed items."""

    def parse(self, text: str) -> list[str]:
        return [item.strip() for item in text.split(",") if item.strip()]


__all__ = [
    "BaseOutputParser",
    "StrOutputParser",
    "JsonOutputParser",
    "CommaSeparatedListOutputParser",
]
