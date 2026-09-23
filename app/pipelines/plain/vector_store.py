"""Qdrant client setup shared by ingestion (writes) and retrieval (reads).

If QDRANT_URL is set, connects to a real Qdrant server (e.g. a Qdrant Cloud
cluster, or the docker-compose one). Otherwise falls back to qdrant-client's
embedded/on-disk mode at QDRANT_PATH — no Docker required for local
development. Same API either way, so switching later is a one-line env var
change, not a code change.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, SparseVectorParams, VectorParams

from app.core import config

# Payload fields filtered on at query time (app/pipelines/plain/retrieval.py's
# cross-reference boost filters on source_doc). Qdrant Cloud rejects a filter
# on a field with no payload index ("Index required but not found") -- the
# embedded/on-disk mode is more permissive and doesn't need this, which is
# exactly why it went unnoticed until switching to a real server. Indexing
# every collection the same way regardless of backend keeps behavior
# consistent instead of only surfacing this the first time someone deploys.
_INDEXED_PAYLOAD_FIELDS = {"source_doc": PayloadSchemaType.KEYWORD}


def get_qdrant_client() -> QdrantClient:
    if config.QDRANT_URL:
        # A remote server's default client timeout (a few seconds) isn't
        # enough for a full-corpus upsert over a slower/higher-latency
        # network path -- observed directly as httpx.WriteTimeout on an
        # otherwise-correct request, not a real server error.
        return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)
    return QdrantClient(path=config.QDRANT_PATH)


def _ensure_payload_indexes(client: QdrantClient, collection_name: str) -> None:
    for field_name, schema in _INDEXED_PAYLOAD_FIELDS.items():
        client.create_payload_index(
            collection_name=collection_name, field_name=field_name, field_schema=schema
        )


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Phase 1 dense-only schema: a single unnamed vector per point."""
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    _ensure_payload_indexes(client, collection_name)


def ensure_hybrid_collection(client: QdrantClient, collection_name: str, dense_size: int) -> None:
    """Phase 3 hybrid schema: named "dense" vector + named "sparse" (BM25) vector,
    combined at query time via RRF fusion."""
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={"dense": VectorParams(size=dense_size, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()},
    )
    _ensure_payload_indexes(client, collection_name)
