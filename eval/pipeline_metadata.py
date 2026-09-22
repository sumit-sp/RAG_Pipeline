"""Builds the full pipeline-config snapshot attached to every Langfuse
experiment run's `metadata`, so the Compare Experiments view (or anyone
reading the run later) can see exactly what produced each run's numbers
without cross-referencing DECISIONS.md/PROGRESS.md by hand.

Run with: python -m eval.pipeline_metadata   (prints the current config, for
a quick sanity check of what a new experiment run would attach)
"""

from app.core import config
from app.pipelines.plain.vector_store import get_qdrant_client


def _embedding_dimension(collection: str | None = None) -> int | None:
    """Reads the dense vector size straight from a live Qdrant collection,
    rather than instantiating the embedding model just to check its dimension.

    Opens its own short-lived client and closes it immediately -- the local
    (embedded, on-disk) Qdrant mode takes an exclusive file lock per client
    instance, so leaving this one open would collide with any other client
    (e.g. a retriever's) the calling script also needs in the same process."""
    collection = collection or config.collection_name()
    client = get_qdrant_client()
    try:
        if not client.collection_exists(collection):
            return None
        info = client.get_collection(collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):  # hybrid: named "dense"/"sparse" vectors
            dense = vectors.get("dense")
            return dense.size if dense else None
        return vectors.size if vectors else None  # dense-only: single unnamed vector
    finally:
        client.close()


def build_pipeline_metadata(overrides: dict | None = None) -> dict:
    """`overrides` lets a caller correct fields that don't survive as live env
    config -- e.g. USE_CONTEXTUAL_HEADERS only affects ingestion, not
    retrieval, so a collection ingested with headers on will show `false`
    here unless the caller passes `{"use_contextual_headers": True}` to
    reflect what's actually baked into the chunks. Also used by the
    historical backfill script, where the *current* live config often
    doesn't match the config a given historical run actually used --
    pass `{"qdrant_collection": "<the collection that run actually used>"}`
    so embedding_dimension is read from the right place."""
    overrides = overrides or {}
    collection = overrides.get("qdrant_collection") or config.collection_name()

    metadata = {
        "chunking_strategy": config.CHUNKING_STRATEGY,
        "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
        "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
        "use_contextual_headers": config.USE_CONTEXTUAL_HEADERS,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
        "use_reranking": config.USE_RERANKING,
        "embedding_backend": config.EMBEDDING_BACKEND,
        "embedding_model": config.LOCAL_EMBEDDING_MODEL,
        "embedding_dimension": _embedding_dimension(collection),
        "generation_model": config.GENERATION_MODEL,
        "qdrant_collection": collection,
    }
    metadata.update(overrides)

    # Conditional fields are decided from the final (post-override) values,
    # not raw live config, so a historical override like retrieval_mode="dense"
    # correctly omits hybrid-only fields.
    if metadata["retrieval_mode"] == "hybrid":
        metadata.setdefault("hybrid_prefetch_limit", config.HYBRID_PREFETCH_LIMIT)
        metadata.setdefault("sparse_embedding_model", "Qdrant/bm25")
    if metadata["use_reranking"]:
        metadata.setdefault("rerank_candidate_limit", config.RERANK_CANDIDATE_LIMIT)
        metadata.setdefault("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    if metadata["use_contextual_headers"]:
        metadata.setdefault("contextual_headers_path", config.CONTEXTUAL_HEADERS_PATH)

    return metadata


if __name__ == "__main__":
    import json

    print(json.dumps(build_pipeline_metadata(), indent=2))
