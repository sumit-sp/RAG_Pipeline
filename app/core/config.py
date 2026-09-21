"""Central place for env-var config, per the "keep all config in env vars" deployment
philosophy in the spec. Loads .env if present (see .env.example)."""

import os

from dotenv import load_dotenv

load_dotenv()

# Pipeline backend selection (Phase 6 adds "langchain" as a second valid value)
PIPELINE_BACKEND = os.environ.get("PIPELINE_BACKEND", "plain")

# Vector store: if QDRANT_URL is set, connect to a real Qdrant server (e.g. the
# docker-compose one). Otherwise fall back to qdrant-client's embedded/on-disk
# mode at QDRANT_PATH — no Docker required. Same client API either way.
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrant_local_data")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "ai_act_corpus")

# Embeddings: local by default (no API key needed). Only "local" is implemented
# in Phase 1 — the Embedder protocol is what makes adding "openai"/"voyage"/
# "cohere" later a new class, not a rewrite.
EMBEDDING_BACKEND = os.environ.get("EMBEDDING_BACKEND", "local")
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Generation: Groq-hosted GPT-OSS models.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "openai/gpt-oss-20b")

# Chunking. "fixed" is the Phase 1 baseline (naive token window). "recursive" is a
# Phase 3 chunk-size-tuning candidate that prefers natural text boundaries.
CHUNK_SIZE_TOKENS = int(os.environ.get("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "50"))
CHUNKING_STRATEGY = os.environ.get("CHUNKING_STRATEGY", "fixed")

# Retrieval — Phase 3 adds "hybrid" (dense + BM25 sparse, RRF fusion) as an
# alternative to Phase 1's "dense"-only baseline. Each mode gets its own Qdrant
# collection (different vector schemas), so switching doesn't require re-ingesting
# whichever mode you're not currently using.
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))
RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")
HYBRID_PREFETCH_LIMIT = int(os.environ.get("HYBRID_PREFETCH_LIMIT", "20"))

# Reranking is an orthogonal post-processing step over whichever RETRIEVAL_MODE is
# active: fetch a wider candidate set, rerank with a cross-encoder, keep the top-k.
USE_RERANKING = os.environ.get("USE_RERANKING", "false").lower() == "true"
RERANK_CANDIDATE_LIMIT = int(os.environ.get("RERANK_CANDIDATE_LIMIT", "20"))


def collection_name() -> str:
    """Each retrieval mode gets its own collection, since dense-only and hybrid
    collections have different vector schemas."""
    return QDRANT_COLLECTION if RETRIEVAL_MODE == "dense" else f"{QDRANT_COLLECTION}_{RETRIEVAL_MODE}"

DATA_RAW_DIR = os.environ.get("DATA_RAW_DIR", "data/raw")
