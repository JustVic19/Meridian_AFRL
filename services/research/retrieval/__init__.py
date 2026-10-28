"""Document retrieval. Embeddings are deterministic and no model is called."""

from research.retrieval.embeddings import (
    EMBEDDING_DIMENSIONS,
    EmbeddingProvider,
    MockEmbeddingProvider,
    assign_embedding,
)
from research.retrieval.retriever import DocumentRetriever, RetrievalHit, RetrievalQuery

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "DocumentRetriever",
    "EmbeddingProvider",
    "MockEmbeddingProvider",
    "RetrievalHit",
    "RetrievalQuery",
    "assign_embedding",
]
