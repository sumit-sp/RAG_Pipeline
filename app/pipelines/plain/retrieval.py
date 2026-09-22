"""Retrieval: dense-only top-k (Phase 1 baseline) or hybrid dense+BM25 with RRF
fusion (Phase 3), selected via RETRIEVAL_MODE.

Phase 3 also adds a targeted cross-reference boost (see EVALUATION_HISTORY.md
Step 9 / DECISIONS.md): corpus-wide ranking systematically buries GDPR chunks
for questions phrased mostly in AI-Act vocabulary, because GDPR is only ~150
of 837 chunks and loses that vocabulary contest even when the right passage is
present. When a question names a regulation from `_CROSS_REFERENCE_TRIGGERS`,
an extra search restricted to just that document (no other content competing)
runs alongside the normal one, and any new chunks it finds are appended."""

from qdrant_client.models import FieldCondition, Filter, FusionQuery, MatchValue, Prefetch

from app.core import config
from app.core.interfaces import Embedder
from app.core.models import Chunk, RetrievedContext
from app.pipelines.plain.embedding import get_embedder
from app.pipelines.plain.reranking import Reranker
from app.pipelines.plain.sparse_embedding import SparseEmbedder
from app.pipelines.plain.vector_store import get_qdrant_client

# keyword (matched case-insensitively as a substring of the question) -> the
# source_doc a supplementary, document-restricted search should target. Only
# GDPR is wired up so far since that's the diagnosed gap (Step 9); a future
# cross-referenced regulation would just add another entry here.
_CROSS_REFERENCE_TRIGGERS = {"gdpr": "adjacent/gdpr_2016_679.html"}


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

    def _search(
        self, query: str, limit: int, doc_filter: Filter | None = None
    ) -> list[RetrievedContext]:
        collection = config.collection_name()

        if config.RETRIEVAL_MODE == "hybrid":
            dense_vector = self.embedder.embed_query(query)
            sparse_vector = self.sparse_embedder.embed_query(query)
            results = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=dense_vector,
                        using="dense",
                        limit=config.HYBRID_PREFETCH_LIMIT,
                        filter=doc_filter,
                    ),
                    Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=config.HYBRID_PREFETCH_LIMIT,
                        filter=doc_filter,
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
                query_filter=doc_filter,
                limit=limit,
            ).points

        return [_to_context(point) for point in results]

    def _cross_reference_boost(
        self, query: str, existing: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        query_lower = query.lower()
        seen_ids = {c.chunk.id for c in existing}
        extra: list[RetrievedContext] = []
        for keyword, source_doc in _CROSS_REFERENCE_TRIGGERS.items():
            if keyword not in query_lower:
                continue
            doc_filter = Filter(
                must=[FieldCondition(key="source_doc", match=MatchValue(value=source_doc))]
            )
            for chunk in self._search(
                query, limit=config.CROSS_REFERENCE_BOOST_LIMIT, doc_filter=doc_filter
            ):
                if chunk.chunk.id not in seen_ids:
                    extra.append(chunk)
                    seen_ids.add(chunk.chunk.id)
        return extra

    def retrieve(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[RetrievedContext]:
        if self.reranker is None:
            results = self._search(query, limit=top_k)
        else:
            candidates = self._search(query, limit=config.RERANK_CANDIDATE_LIMIT)
            results = self.reranker.rerank(query, candidates, top_k=top_k)

        if not config.USE_CROSS_REFERENCE_BOOST:
            return results
        return results + self._cross_reference_boost(query, results)
