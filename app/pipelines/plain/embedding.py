"""Concrete Embedder implementations for the plain backend.

Two local (no API key) backends, selected via EMBEDDING_BACKEND:
- "local" (default, unchanged since Phase 1): sentence-transformers -- pulls
  in PyTorch. Kept as-is; nothing about this path changes.
- "fastembed" (added to fit Render's free-tier 512MB RAM limit -- see
  PROGRESS.md/DECISIONS.md's Render-OOM entry): runs the *same*
  BAAI/bge-small-en-v1.5 weights via fastembed's ONNX runtime instead of
  PyTorch. fastembed was already a required dependency (sparse_embedding.py's
  BM25 model), so this backend adds no new dependency -- it just stops a
  second heavyweight ML runtime (PyTorch) from being loaded alongside
  onnxruntime, which is what was pushing memory to the free tier's ceiling.
  Not re-validated against the eval harness (explicit user instruction) --
  same model weights, same query-prefix handling, swapped runtime only.

Adding a hosted provider later means adding a new class here that satisfies
the same Embedder protocol and wiring it into get_embedder(); nothing else
changes.
"""

from app.core import config
from app.core.interfaces import Embedder

# BGE models are trained to expect this instruction prefix on queries (not on the
# documents being searched) — leaving it off measurably hurts retrieval quality.
# fastembed's query_embed() does NOT apply this automatically for dense text
# models (only its BM25 sparse model has a query-specific prefix built in --
# see sparse_embedding.py) -- confirmed directly, not assumed, so both
# backends below apply it explicitly and identically.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    """Runs a sentence-transformers model on-device. No API key, no network calls."""

    def __init__(self, model_name: str = config.LOCAL_EMBEDDING_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            _BGE_QUERY_PREFIX + text, normalize_embeddings=True, show_progress_bar=False
        )
        return vector.tolist()


class FastEmbedEmbedder:
    """Runs the same BAAI/bge-small-en-v1.5 weights via fastembed's ONNX
    runtime -- no PyTorch. Output is already unit-normalized (verified
    directly: ||v|| ~= 1.0), matching LocalEmbedder's explicit normalization,
    so the two are drop-in equivalent from the retriever's perspective."""

    def __init__(self, model_name: str = config.LOCAL_EMBEDDING_MODEL):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self.model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        vector = next(iter(self.model.query_embed(_BGE_QUERY_PREFIX + text)))
        return vector.tolist()


def get_embedder() -> Embedder:
    if config.EMBEDDING_BACKEND == "local":
        return LocalEmbedder()
    if config.EMBEDDING_BACKEND == "fastembed":
        return FastEmbedEmbedder()
    raise NotImplementedError(
        f"EMBEDDING_BACKEND={config.EMBEDDING_BACKEND!r} is not implemented yet — "
        "'local' and 'fastembed' exist so far (see DECISIONS.md)."
    )
