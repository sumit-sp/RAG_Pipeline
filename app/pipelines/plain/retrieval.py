"""Retrieval: dense-only top-k (Phase 1 baseline) or hybrid dense+BM25 with RRF
fusion (Phase 3), selected via RETRIEVAL_MODE."""

from qdrant_client.models import FusionQuery, Prefetch

from app.core import config
from app.core.interfaces import Embedder
from app.core.models import Chunk, RetrievedContext
from app.pipelines.plain.embedding import get_embedder
from app.pipelines.plain.reranking import Reranker
from app.pipelines.plain.sparse_embedding import SparseEmbedder
from app.pipelines.plain.vector_store import get_qdrant_client


def _to_context(point) -> RetrievedContext:
    return RetrievedContext(
        chunk=Chunk(
            id=str(point.id),
            text=point.payload["text"],
            source_doc=point.payload["source_doc"],
            doc_type=point.payload["doc_type"],
            chunk_index=point.payload["chunk_index"],
        ),
        score=point.score,
    )


class PlainRetriever:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self.sparse_embedder = SparseEmbedder() if config.RETRIEVAL_MODE == "hybrid" else None
        self.reranker = Reranker() if config.USE_RERANKING else None
        self.client = get_qdrant_client()

    def _search(self, query: str, limit: int) -> list[RetrievedContext]:
        collection = config.collection_name()

        if config.RETRIEVAL_MODE == "hybrid":
            dense_vector = self.embedder.embed_query(query)
            sparse_vector = self.sparse_embedder.embed_query(query)
            results = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector, using="dense", limit=config.HYBRID_PREFETCH_LIMIT
                    ),
                    Prefetch(
                        query=sparse_vector, using="sparse", limit=config.HYBRID_PREFETCH_LIMIT
                    ),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=limit,
            ).points
        else:
            query_vector = self.embedder.embed_query(query)
            results = self.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit,
            ).points

        return [_to_context(point) for point in results]

    def retrieve(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[RetrievedContext]:
        if self.reranker is None:
            return self._search(query, limit=top_k)

        candidates = self._search(query, limit=config.RERANK_CANDIDATE_LIMIT)
        return self.reranker.rerank(query, candidates, top_k=top_k)
