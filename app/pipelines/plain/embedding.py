"""Concrete Embedder implementations for the plain backend.

Only "local" is implemented in Phase 1 (no API key required — see DECISIONS.md).
Adding a hosted provider later means adding a new class here that satisfies the
same Embedder protocol and wiring it into get_embedder(); nothing else changes.
"""

from sentence_transformers import SentenceTransformer

from app.core import config
from app.core.interfaces import Embedder

# BGE models are trained to expect this instruction prefix on queries (not on the
# documents being searched) — leaving it off measurably hurts retrieval quality.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    """Runs a sentence-transformers model on-device. No API key, no network calls."""

    def __init__(self, model_name: str = config.LOCAL_EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            _BGE_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()


def get_embedder() -> Embedder:
    if config.EMBEDDING_BACKEND == "local":
        return LocalEmbedder()
    raise NotImplementedError(
        f"EMBEDDING_BACKEND={config.EMBEDDING_BACKEND!r} is not implemented yet — "
        "only 'local' exists so far (see DECISIONS.md)."
    )
