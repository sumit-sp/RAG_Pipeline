"""Cross-encoder reranking (Phase 3): re-scores a larger candidate set retrieved by
whatever RETRIEVAL_MODE is active, and returns the top-k by rerank score.

Local model (sentence-transformers CrossEncoder) — no API key, same "local first"
pattern as the rest of this project's models.
"""

from sentence_transformers import CrossEncoder

from app.core.models import RetrievedContext

_RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = _RERANK_MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedContext], top_k: int
    ) -> list[RetrievedContext]:
        if not candidates:
            return candidates
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)
        reranked = sorted(
            (RetrievedContext(chunk=c.chunk, score=float(s)) for c, s in zip(candidates, scores)),
            key=lambda c: c.score,
            reverse=True,
        )
        return reranked[:top_k]
