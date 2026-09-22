"""Retrieval: dense-only top-k (Phase 1 baseline) or hybrid dense+BM25 with RRF
fusion (Phase 3), selected via RETRIEVAL_MODE.

Phase 3 also adds a cross-reference boost (see EVALUATION_HISTORY.md Steps 9,
11, 12 / DECISIONS.md): corpus-wide ranking systematically buries a small
minority document (e.g. GDPR is only ~150 of 837 chunks) even when its
content is the right answer, because it's competing against the whole corpus'
vocabulary at once. Two independent signals can trigger a supplementary,
document-restricted search (no other content competing) for a chunk any of
these signals name:
  1. The question itself names a cross-referenced document (via
     `detect_references` on the query text) — cheap, but depends on the
     user's exact phrasing (Step 11).
  2. A chunk already retrieved explicitly names another document that isn't
     otherwise represented (via each chunk's `references` payload, tagged at
     ingestion time) — generalizes beyond question phrasing, since it reacts
     to what the documents themselves say, not how the question is worded
     (Step 12).
Both use the same alias registry in cross_references.py, so there's one
source of truth for what counts as a "named" cross-reference."""

from qdrant_client.models import FieldCondition, Filter, FusionQuery, MatchValue, Prefetch

from app.core import config
from app.core.interfaces import Embedder
from app.core.models import Chunk, RetrievedContext
from app.pipelines.plain.cross_references import detect_references
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
            references=point.payload.get("references", []),
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
        existing_docs = {c.chunk.source_doc for c in existing}
        seen_ids = {c.chunk.id for c in existing}

        targets: set[str] = set(detect_references(query))  # signal 1: question wording
        for c in existing:  # signal 2: what the retrieved chunks themselves name
            targets.update(c.chunk.references)

        extra: list[RetrievedContext] = []
        for source_doc in targets:
            if source_doc in existing_docs:
                continue  # already represented in this result set; no boost needed
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
