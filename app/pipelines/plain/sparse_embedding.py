"""BM25-style sparse embeddings for hybrid search (Phase 3).

Uses fastembed's local Qdrant/bm25 model — no API key, consistent with the rest of
this project's "local first" embedding choices (see DECISIONS.md). Sparse vectors
are combined with the existing dense (sentence-transformers) vectors at query time
via Qdrant's native RRF fusion — see vector_store.py and retrieval.py.
"""

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

_SPARSE_MODEL_NAME = "Qdrant/bm25"


class SparseEmbedder:
    def __init__(self, model_name: str = _SPARSE_MODEL_NAME):
        self.model = SparseTextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[SparseVector]:
        return [
            SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self.model.embed(texts)
        ]

    def embed_query(self, text: str) -> SparseVector:
        e = next(iter(self.model.query_embed(text)))
        return SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
