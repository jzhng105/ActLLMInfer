"""Message primitives, mirroring the LangChain Core message hierarchy.

Every chat-model call in actllminfer accepts a sequence of ``BaseMessage``
objects (or anything ``coerce_messages`` knows how to convert).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence, Union


MessageContent = Union[str, list[Union[str, dict]]]
"""Text, or a list of multimodal parts ({"type": "text", "text": "..."} etc.)."""


@dataclass
class BaseMessage:
    """Common base for all message types."""

    content: MessageContent
    additional_kwargs: dict = field(default_factory=dict)
    name: str | None = None

    #: subclasses override this
    type: str = field(default="base", init=False, repr=False)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"role": self.type, "content": self.content}
        if self.name:
            out["name"] = self.name
        if self.additional_kwargs:
            out.update(self.additional_kwargs)
        return out

    def __post_init__(self) -> None:
        if isinstance(self.content, list):
            for part in self.content:
                if not isinstance(part, (str, dict)):
                    raise TypeError(
                        f"Message content parts must be str or dict, got {type(part).__name__}"
                    )


@dataclass
class SystemMessage(BaseMessage):
    type: str = field(default="system", init=False, repr=False)


@dataclass
class HumanMessage(BaseMessage):
    type: str = field(default="user", init=False, repr=False)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["role"] = "user"
        return d


@dataclass
class AIMessage(BaseMessage):
    """Assistant response. May carry ``tool_calls`` for function/tool calling."""

    tool_calls: list["ToolCall"] = field(default_factory=list)
    type: str = field(default="assistant", init=False, repr=False)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["role"] = "assistant"
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d


@dataclass
class ToolMessage(BaseMessage):
    """Result of a tool call, fed back to the model."""

    tool_call_id: str = ""
    type: str = field(default="tool", init=False, repr=False)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["role"] = "tool"
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class FunctionMessage(BaseMessage):
    """Legacy OpenAI ``function`` role; used by some providers."""

    type: str = field(default="function", init=False, repr=False)


@dataclass
class ToolCall:
    """Structured representation of a single tool/function call."""

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        import json as _json

        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": _json.dumps(self.args, ensure_ascii=False),
            },
        }


# ---------------------------------------------------------------------------
# Chunk variants used during streaming
# ---------------------------------------------------------------------------


@dataclass
class AIMessageChunk(AIMessage):
    """Partial assistant message produced during ``stream``.

    Chunks support ``+`` so callers can fold a stream into a final message::

        final = None
        for chunk in llm.stream(messages):
            final = chunk if final is None else final + chunk
    """

    type: str = field(default="assistant", init=False, repr=False)

    def __add__(self, other: "AIMessageChunk") -> "AIMessageChunk":
        if not isinstance(other, AIMessageChunk):
            return NotImplemented
        merged_content: MessageContent
        if isinstance(self.content, str) and isinstance(other.content, str):
            merged_content = self.content + other.content
        else:
            left = self.content if isinstance(self.content, list) else [self.content]
            right = other.content if isinstance(other.content, list) else [other.content]
            merged_content = [*left, *right]
        merged_kwargs = {**self.additional_kwargs, **other.additional_kwargs}
        merged_calls = _merge_tool_calls(self.tool_calls, other.tool_calls)
        return AIMessageChunk(
            content=merged_content,
            additional_kwargs=merged_kwargs,
            tool_calls=merged_calls,
            name=self.name or other.name,
        )


def _merge_tool_calls(left: list[ToolCall], right: list[ToolCall]) -> list[ToolCall]:
    if not left:
        return list(right)
    if not right:
        return list(left)
    by_id: dict[str, ToolCall] = {tc.id: ToolCall(tc.id, tc.name, dict(tc.args)) for tc in left}
    for tc in right:
        if tc.id in by_id:
            existing = by_id[tc.id]
            existing.name = existing.name or tc.name
            existing.args.update(tc.args)
        else:
            by_id[tc.id] = ToolCall(tc.id, tc.name, dict(tc.args))
    return list(by_id.values())


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

MessageLike = Union[
    BaseMessage,
    str,
    tuple,  # ("system", "you are ...") / ("user", "hi")
    Mapping[str, Any],  # {"role": "user", "content": "hi"}
]


_ROLE_TO_CLASS: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
    "tool": ToolMessage,
    "function": FunctionMessage,
}


def coerce_message(msg: MessageLike) -> BaseMessage:
    """Convert one of the accepted forms into a ``BaseMessage``."""
    if isinstance(msg, BaseMessage):
        return msg
    if isinstance(msg, str):
        return HumanMessage(content=msg)
    if isinstance(msg, tuple):
        if len(msg) != 2:
            raise ValueError(f"Tuple messages must be (role, content); got length {len(msg)}")
        role, content = msg
        cls = _ROLE_TO_CLASS.get(str(role).lower())
        if cls is None:
            raise ValueError(f"Unknown message role: {role!r}")
        return cls(content=content)
    if isinstance(msg, Mapping):
        role = str(msg.get("role", "user")).lower()
        cls = _ROLE_TO_CLASS.get(role)
        if cls is None:
            raise ValueError(f"Unknown message role: {role!r}")
        kwargs: dict[str, Any] = {"content": msg.get("content", "")}
        if "name" in msg:
            kwargs["name"] = msg["name"]
        if cls is ToolMessage and "tool_call_id" in msg:
            kwargs["tool_call_id"] = msg["tool_call_id"]
        if cls is AIMessage and msg.get("tool_calls"):
            kwargs["tool_calls"] = [
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name") or tc.get("name", ""),
                    args=_parse_tool_args(tc),
                )
                for tc in msg["tool_calls"]
            ]
        return cls(**kwargs)
    raise TypeError(f"Cannot coerce {type(msg).__name__} into a BaseMessage")


def _parse_tool_args(tc: Mapping[str, Any]) -> dict:
    import json as _json

    fn = tc.get("function") or {}
    raw = fn.get("arguments", tc.get("arguments", {}))
    if isinstance(raw, str):
        try:
            return _json.loads(raw) if raw else {}
        except _json.JSONDecodeError:
            return {"_raw": raw}
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def coerce_messages(messages: Union[str, MessageLike, Sequence[MessageLike]]) -> list[BaseMessage]:
    """Normalize the public ``invoke``/``stream`` input."""
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]
    if isinstance(messages, BaseMessage):
        return [messages]
    if isinstance(messages, Mapping):
        return [coerce_message(messages)]
    if isinstance(messages, Iterable):
        return [coerce_message(m) for m in messages]
    raise TypeError(f"Cannot coerce {type(messages).__name__} into messages")


__all__ = [
    "BaseMessage",
    "SystemMessage",
    "HumanMessage",
    "AIMessage",
    "AIMessageChunk",
    "ToolMessage",
    "FunctionMessage",
    "ToolCall",
    "MessageLike",
    "coerce_message",
    "coerce_messages",
]
