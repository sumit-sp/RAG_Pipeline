"""Phase 1 retrieval: top-k dense vector search only — no hybrid, no reranking."""

from app.core import config
from app.core.interfaces import Embedder
from app.core.models import Chunk, RetrievedContext
from app.pipelines.plain.embedding import get_embedder
from app.pipelines.plain.vector_store import get_qdrant_client


class PlainRetriever:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self.client = get_qdrant_client()

    def retrieve(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> list[RetrievedContext]:
        query_vector = self.embedder.embed_query(query)
        results = self.client.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            RetrievedContext(
                chunk=Chunk(
                    id=str(point.id),
                    text=point.payload["text"],
                    source_doc=point.payload["source_doc"],
                    doc_type=point.payload["doc_type"],
                    chunk_index=point.payload["chunk_index"],
                ),
                score=point.score,
            )
            for point in results
        ]
