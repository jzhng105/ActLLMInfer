"""Embedding model abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEmbeddings(ABC):
    """Minimal embedding interface."""

    @abstractmethod
    def embed_documents(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        """Return embeddings for a batch of documents."""

    @abstractmethod
    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Return an embedding for a single query text."""
