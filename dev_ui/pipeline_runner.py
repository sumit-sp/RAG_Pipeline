"""Reusable retrieve+generate logic for the local experimentation UI.

Not part of the production pipeline. A thin layer over the real
PlainRetriever/PlainGenerator that adds the introspection this UI needs
(which retrieval path found each chunk, per-chunk component scores, the
exact prompt sent, timing, token usage) without changing production code.

Reaches into a couple of PlainRetriever's underscore-prefixed methods
(_search, _cross_reference_boost) and generation's private prompt-building
helpers on purpose, to tag/display exactly what the real pipeline does
rather than approximate it. If retrieval.py's or generation.py's internals
are restructured, this file needs updating alongside them.
"""

import time
from dataclasses import dataclass

from app.core import config
from app.core.models import RetrievedContext
from app.pipelines.plain import generation as generation_module
from app.pipelines.plain.generation import PlainGenerator
from app.pipelines.plain.retrieval import PlainRetriever


@dataclass
class ScoredChunk:
    context: RetrievedContext
    origin: str  # "top-k" or "cross-reference boost"
    dense_score: float | None = None
    sparse_score: float | None = None


@dataclass
class RunResult:
    question: str
    chunks: list[ScoredChunk]
    answer_text: str
    citations: list[str]
    prompt_system: str
    prompt_user: str
    usage: dict | None
    retrieval_seconds: float
    generation_seconds: float
    generation_model: str
    retrieval_mode: str
    used_boost: bool
    used_reranking: bool
    top_k: int
    collection: str


def retriever_cache_key() -> tuple:
    """Everything that changes what PlainRetriever() actually builds/loads --
    pass this as the cache key so the UI only pays a reload cost when one of
    these genuinely changed, not on every query."""
    return (
        config.QDRANT_URL,
        config.QDRANT_PATH,
        config.RETRIEVAL_MODE,
        config.USE_RERANKING,
        config.EMBEDDING_BACKEND,
        config.LOCAL_EMBEDDING_MODEL,
    )


def _component_scores(
    retriever: PlainRetriever, query: str, chunk_ids: set[str]
) -> tuple[dict[str, float], dict[str, float]]:
    """Best-effort dense/sparse component scores for chunks the hybrid fusion
    already returned -- two extra display-only queries, never used to change
    which chunks are actually retrieved. Only meaningful in hybrid mode."""
    dense_scores: dict[str, float] = {}
    sparse_scores: dict[str, float] = {}
    if config.RETRIEVAL_MODE != "hybrid" or retriever.sparse_embedder is None or not chunk_ids:
        return dense_scores, sparse_scores

    collection = config.collection_name()
    limit = max(config.HYBRID_PREFETCH_LIMIT, len(chunk_ids))

    dense_vector = retriever.embedder.embed_query(query)
    for point in retriever.client.query_points(
        collection_name=collection, query=dense_vector, using="dense", limit=limit
    ).points:
        if str(point.id) in chunk_ids:
            dense_scores[str(point.id)] = point.score

    sparse_vector = retriever.sparse_embedder.embed_query(query)
    for point in retriever.client.query_points(
        collection_name=collection, query=sparse_vector, using="sparse", limit=limit
    ).points:
        if str(point.id) in chunk_ids:
            sparse_scores[str(point.id)] = point.score

    return dense_scores, sparse_scores


def run_pipeline(retriever: PlainRetriever, generator: PlainGenerator, question: str, top_k: int) -> RunResult:
    t0 = time.perf_counter()
    if retriever.reranker is None:
        primary = retriever._search(question, limit=top_k)
    else:
        candidates = retriever._search(question, limit=config.RERANK_CANDIDATE_LIMIT)
        primary = retriever.reranker.rerank(question, candidates, top_k=top_k)

    boost: list[RetrievedContext] = []
    if config.USE_CROSS_REFERENCE_BOOST:
        boost = retriever._cross_reference_boost(question, primary)
    retrieval_seconds = time.perf_counter() - t0

    all_ids = {c.chunk.id for c in primary} | {c.chunk.id for c in boost}
    dense_scores, sparse_scores = _component_scores(retriever, question, all_ids)

    def _tag(c: RetrievedContext, origin: str) -> ScoredChunk:
        return ScoredChunk(
            context=c,
            origin=origin,
            dense_score=dense_scores.get(c.chunk.id),
            sparse_score=sparse_scores.get(c.chunk.id),
        )

    chunks = [_tag(c, "top-k") for c in primary] + [_tag(c, "cross-reference boost") for c in boost]
    contexts = primary + boost

    t1 = time.perf_counter()
    answer = generator.generate(question, contexts)
    generation_seconds = time.perf_counter() - t1

    formatted_context = generation_module._format_context(contexts)
    prompt_user = f"Context excerpts:\n{formatted_context}\n\nQuestion: {question}"

    return RunResult(
        question=question,
        chunks=chunks,
        answer_text=answer.text,
        citations=answer.citations,
        prompt_system=generation_module._SYSTEM_PROMPT,
        prompt_user=prompt_user,
        usage=answer.usage,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=generation_seconds,
        generation_model=config.GENERATION_MODEL,
        retrieval_mode=config.RETRIEVAL_MODE,
        used_boost=config.USE_CROSS_REFERENCE_BOOST,
        used_reranking=config.USE_RERANKING,
        top_k=top_k,
        collection=config.collection_name(),
    )
