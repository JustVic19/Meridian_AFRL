"""Replaceable document embeddings. The mock provider does not call a model."""

import hashlib
import math
import re
from typing import Protocol

from sqlalchemy.orm import Session

from api.models import Document

EMBEDDING_DIMENSIONS = 32
_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingError(Exception):
    """The text cannot be embedded or the provider width does not match."""


class EmbeddingProvider(Protocol):
    @property
    def dimensions(self) -> int:
        """Width of every vector this provider returns."""

    def embed(self, text: str) -> tuple[float, ...]:
        """Return one deterministic vector for this text."""


class MockEmbeddingProvider:
    """Hash tokens into a fixed vector. The same text always returns the same value."""

    def __init__(self, dimensions: int = EMBEDDING_DIMENSIONS) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> tuple[float, ...]:
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            raise EmbeddingError("text has no tokens")
        vector = [0.0] * self._dimensions
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:2], "big") % self._dimensions
            sign = 1.0 if digest[2] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            raise EmbeddingError("embedding norm is zero")
        return tuple(value / norm for value in vector)


def assign_embedding(session: Session, document: Document, provider: EmbeddingProvider) -> None:
    """Write one vector onto the document. A second call updates that same column."""
    if provider.dimensions != EMBEDDING_DIMENSIONS:
        raise EmbeddingError("embedding provider width does not match the document column")
    if document.content is None:
        raise EmbeddingError("document has no content")
    document.embedding = list(provider.embed(document.content))
    session.flush()
