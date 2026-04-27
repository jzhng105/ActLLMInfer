"""Result containers returned from chat models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .messages import AIMessage, AIMessageChunk


@dataclass
class ChatGeneration:
    """A single completion produced by a chat model."""

    message: AIMessage
    generation_info: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        c = self.message.content
        if isinstance(c, str):
            return c
        return "".join(part if isinstance(part, str) else part.get("text", "") for part in c)


@dataclass
class ChatGenerationChunk:
    message: AIMessageChunk
    generation_info: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        c = self.message.content
        if isinstance(c, str):
            return c
        return "".join(part if isinstance(part, str) else part.get("text", "") for part in c)


@dataclass
class ChatResult:
    """Full response from a chat model: one or more generations + provider metadata."""

    generations: list[ChatGeneration]
    llm_output: dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> AIMessage:
        if not self.generations:
            raise ValueError("ChatResult contains no generations")
        return self.generations[0].message

    @property
    def text(self) -> str:
        return self.generations[0].text if self.generations else ""


__all__ = ["ChatGeneration", "ChatGenerationChunk", "ChatResult"]
