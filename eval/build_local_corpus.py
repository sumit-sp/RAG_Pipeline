"""One-time (per change) build of a local, on-disk Qdrant corpus -- no network,
no Qdrant Cloud -- so retrieval/decomposition experiments can run while Cloud
connectivity is down (see PROGRESS.md's Step 22/23 network notes).

Mirrors current production config (.env / DEPLOYMENT.md): hybrid retrieval,
recursive chunking, contextual headers projected onto recursive chunks
(eval/contextual_headers_recursive.jsonl), cross-reference boost on. Only the
storage location changes -- qdrant-client's embedded mode at
eval/local_qdrant_data/ instead of the real cluster.

Re-run this whenever data/raw/ or the header/chunking config changes; it's
idempotent per collection name (PlainIngestor.ensure_collection is a no-op if
the collection already exists) but does NOT re-embed on repeat runs unless
you delete eval/local_qdrant_data/ first -- delete it if you want a clean
rebuild rather than an incremental one.

Run with: python -m eval.build_local_corpus
"""

from pathlib import Path

from app.core import config
from app.pipelines.plain.ingestion import PlainIngestor
from eval.experiment_lib import use_local_qdrant

# Same base name production uses, so eval/experiment_lib.py's set_corpus()
# names line up whether QDRANT_URL is live or not -- only the storage backend
# differs (embedded on-disk local_qdrant_data/ vs. the real Cloud cluster).
CORPUS_BASE_NAME = "ai_act_corpus_recursive_ctxheaders"


def main() -> None:
    use_local_qdrant()
    config.QDRANT_COLLECTION = CORPUS_BASE_NAME
    config.CHUNKING_STRATEGY = "recursive"
    config.USE_CONTEXTUAL_HEADERS = True
    config.CONTEXTUAL_HEADERS_PATH = "eval/contextual_headers_recursive.jsonl"
    # RETRIEVAL_MODE ("hybrid"), USE_CROSS_REFERENCE_BOOST (true) and
    # USE_RERANKING (false) already default to production's values.

    count = PlainIngestor().ingest(Path(config.DATA_RAW_DIR))
    print(
        f"Indexed {count} chunks from {config.DATA_RAW_DIR} into local "
        f"collection '{config.collection_name()}' at {config.QDRANT_PATH}"
    )


if __name__ == "__main__":
    main()
