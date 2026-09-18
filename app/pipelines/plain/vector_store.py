"""Qdrant client setup shared by ingestion (writes) and retrieval (reads).

If QDRANT_URL is set, connects to a real Qdrant server (e.g. the docker-compose
one). Otherwise falls back to qdrant-client's embedded/on-disk mode at
QDRANT_PATH — no Docker required for local development. Same API either way,
so switching later is a one-line env var change, not a code change.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core import config


def get_qdrant_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)
    return QdrantClient(path=config.QDRANT_PATH)


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
