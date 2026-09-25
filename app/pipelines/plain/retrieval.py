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

from typing import Any

from qdrant_client.models import FieldCondition, Filter, FusionQuery, MatchValue, Prefetch

from app.core import config
from app.core.interfaces import Embedder
from app.core.models import Chunk, RetrievedContext
from app.core.tracing import trace_span
from app.pipelines.plain.cross_references import detect_references
from app.pipelines.plain.embedding import get_embedder
from app.pipelines.plain.query_decomposition import QueryDecomposer
from app.pipelines.plain.reranking import Reranker
from app.pipelines.plain.sparse_embedding import SparseEmbedder
from app.pipelines.plain.vector_store import get_qdrant_client


def _merge_dedupe(context_lists: list[list[RetrievedContext]]) -> list[RetrievedContext]:
    """Combines retrieval runs for multiple (sub-)queries into one list,
    keeping first-seen order and dropping repeats -- same merge Step 19/23
    validated offline (eval/query_decomposition_experiment.py)."""
    seen_ids: set[str] = set()
    merged: list[RetrievedContext] = []
    for contexts in context_lists:
        for c in contexts:
            if c.chunk.id not in seen_ids:
                merged.append(c)
                seen_ids.add(c.chunk.id)
    return merged


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
        self.decomposer = QueryDecomposer() if config.USE_QUERY_DECOMPOSITION else None
        self.client = get_qdrant_client()

    def _search(
        self, query: str, limit: int, doc_filter: Filter | None = None
    ) -> list[RetrievedContext]:
        collection = config.collection_name()

        with trace_span(
            "embed-query",
            as_type="span",
            input={"query": query, "mode": config.RETRIEVAL_MODE},
        ) as embed_trace:
            if config.RETRIEVAL_MODE == "hybrid":
                dense_vector = self.embedder.embed_query(query)
                sparse_vector = self.sparse_embedder.embed_query(query)
            else:
                dense_vector = self.embedder.embed_query(query)
                sparse_vector = None
            embed_trace.set_output({"dense_dims": len(dense_vector)})

        with trace_span(
            "qdrant-search",
            as_type="retriever",
            input={"collection": collection, "limit": limit, "filtered": doc_filter is not None},
        ) as search_trace:
            if config.RETRIEVAL_MODE == "hybrid":
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
                results = self.client.query_points(
                    collection_name=collection,
                    query=dense_vector,
                    query_filter=doc_filter,
                    limit=limit,
                ).points
            search_trace.set_output({"hits": len(results)})

        return [_to_context(point) for point in results]

    def _cross_reference_boost(
        self, query: str, existing: list[RetrievedContext]
    ) -> list[RetrievedContext]:
        with trace_span("cross-reference-boost", as_type="span", input={"query": query}) as boost_trace:
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
            boost_trace.set_output({"targets": sorted(targets), "extra_chunks": len(extra)})
        return extra

    def _retrieve_single(self, query: str, top_k: int) -> list[RetrievedContext]:
        with trace_span("retrieve-single", as_type="retriever", input={"query": query, "top_k": top_k}) as single_trace:
            if self.reranker is None:
                results = self._search(query, limit=top_k)
            else:
                candidates = self._search(query, limit=config.RERANK_CANDIDATE_LIMIT)
                results = self.reranker.rerank(query, candidates, top_k=top_k)

            if config.USE_CROSS_REFERENCE_BOOST:
                results = results + self._cross_reference_boost(query, results)

            single_trace.set_output({"chunks": len(results)})
        return results

    def retrieve(
        self, query: str, top_k: int = config.RETRIEVAL_TOP_K, client: Any | None = None
    ) -> list[RetrievedContext]:
        # A single-hop question (or decomposition off/failed) decomposes to
        # [query] -- one _retrieve_single call, identical to pre-Step-24
        # behavior. A multi-hop question retrieves once per sub-question
        # (each at the same top_k, not a smaller per-sub-question budget --
        # matches what Step 19/23 actually validated) and merges+dedupes.
        # `client`, when given (e.g. a byok-resolved, per-request Groq
        # client), is only used for the decomposition call -- retrieval
        # itself never calls Groq -- and is never stored on self.decomposer,
        # so it can't leak into a later call that omits it.
        with trace_span("retrieval", as_type="retriever", input={"question": query, "top_k": top_k}) as retrieval_trace:
            sub_queries = self.decomposer.decompose(query, client=client) if self.decomposer else [query]
            merged = _merge_dedupe([self._retrieve_single(sq, top_k) for sq in sub_queries])
            retrieval_trace.set_output(
                {"sub_queries": sub_queries, "merged_chunks": len(merged)}
            )
        return merged
