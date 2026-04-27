"""Prompt templates: ``PromptTemplate`` for strings, ``ChatPromptTemplate`` for messages.

Templates use ``str.format`` placeholders (``{var}``), keeping behavior aligned
with langchain_core. Variables that are not present are surfaced as a
``KeyError`` at format time so misuse is loud.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence, Union

from .messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    MessageLike,
    SystemMessage,
    ToolMessage,
    coerce_message,
)
from .runnables import Runnable


def _extract_variables(template: str) -> list[str]:
    formatter = string.Formatter()
    seen: list[str] = []
    for _, field_name, _, _ in formatter.parse(template):
        if field_name and field_name not in seen:
            # str.format supports "name", "name.attr", "name[0]"; we only want the root.
            root = field_name.split(".")[0].split("[")[0]
            if root and root not in seen:
                seen.append(root)
    return seen


@dataclass
class PromptTemplate(Runnable[dict, str]):
    """Single-string prompt template."""

    template: str
    input_variables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.input_variables:
            self.input_variables = _extract_variables(self.template)

    @classmethod
    def from_template(cls, template: str) -> "PromptTemplate":
        return cls(template=template)

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

    def invoke(self, input: dict[str, Any], **kwargs: Any) -> str:
        return self.format(**input)


# ---------------------------------------------------------------------------
# Chat prompts
# ---------------------------------------------------------------------------


_ROLE_TO_CLASS = {
    "system": SystemMessage,
    "user": HumanMessage,
    "human": HumanMessage,
    "assistant": AIMessage,
    "ai": AIMessage,
    "tool": ToolMessage,
}


@dataclass
class _MessageTemplate:
    role: str
    template: str

    def format(self, variables: dict[str, Any]) -> BaseMessage:
        text = self.template.format(**variables)
        cls = _ROLE_TO_CLASS[self.role]
        return cls(content=text)

    @property
    def input_variables(self) -> list[str]:
        return _extract_variables(self.template)


@dataclass
class MessagesPlaceholder:
    """Splice in a list of pre-built messages at template format time.

    Example::

        ChatPromptTemplate.from_messages([
            ("system", "You are helpful."),
            MessagesPlaceholder("history"),
            ("user", "{question}"),
        ])
    """

    variable_name: str
    optional: bool = False


ChatTemplateItem = Union[_MessageTemplate, MessagesPlaceholder, BaseMessage]


@dataclass
class ChatPromptTemplate(Runnable[dict, list[BaseMessage]]):
    messages: list[ChatTemplateItem] = field(default_factory=list)
    input_variables: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.input_variables:
            self.input_variables = self._collect_variables()

    def _collect_variables(self) -> list[str]:
        seen: list[str] = []
        for item in self.messages:
            if isinstance(item, _MessageTemplate):
                for v in item.input_variables:
                    if v not in seen:
                        seen.append(v)
            elif isinstance(item, MessagesPlaceholder):
                if item.variable_name not in seen:
                    seen.append(item.variable_name)
        return seen

    @classmethod
    def from_messages(cls, messages: Sequence) -> "ChatPromptTemplate":
        items: list[ChatTemplateItem] = []
        for raw in messages:
            if isinstance(raw, MessagesPlaceholder):
                items.append(raw)
            elif isinstance(raw, BaseMessage):
                items.append(raw)
            elif isinstance(raw, tuple) and len(raw) == 2:
                role, template = raw
                role_key = str(role).lower()
                if role_key not in _ROLE_TO_CLASS:
                    raise ValueError(f"Unknown role in chat template: {role!r}")
                items.append(_MessageTemplate(role=role_key, template=str(template)))
            elif isinstance(raw, str):
                # Bare string == user message.
                items.append(_MessageTemplate(role="user", template=raw))
            else:
                raise TypeError(f"Unsupported chat-template item: {type(raw).__name__}")
        return cls(messages=items)

    def format_messages(self, **variables: Any) -> list[BaseMessage]:
        out: list[BaseMessage] = []
        for item in self.messages:
            if isinstance(item, _MessageTemplate):
                out.append(item.format(variables))
            elif isinstance(item, MessagesPlaceholder):
                if item.variable_name not in variables:
                    if item.optional:
                        continue
                    raise KeyError(
                        f"MessagesPlaceholder {item.variable_name!r} not provided"
                    )
                value = variables[item.variable_name]
                if isinstance(value, BaseMessage):
                    out.append(value)
                elif isinstance(value, Iterable):
                    for m in value:
                        out.append(coerce_message(m))
                else:
                    raise TypeError(
                        f"MessagesPlaceholder {item.variable_name!r} must be a message "
                        f"or sequence of messages, got {type(value).__name__}"
                    )
            elif isinstance(item, BaseMessage):
                out.append(item)
        return out

    def invoke(self, input: dict[str, Any], **kwargs: Any) -> list[BaseMessage]:
        return self.format_messages(**input)


__all__ = [
    "PromptTemplate",
    "ChatPromptTemplate",
    "MessagesPlaceholder",
    "MessageLike",
]
