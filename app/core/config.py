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

# Chunking (Phase 1: fixed-size, no cleverness)
CHUNK_SIZE_TOKENS = int(os.environ.get("CHUNK_SIZE_TOKENS", "500"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("CHUNK_OVERLAP_TOKENS", "50"))

# Retrieval
RETRIEVAL_TOP_K = int(os.environ.get("RETRIEVAL_TOP_K", "5"))

DATA_RAW_DIR = os.environ.get("DATA_RAW_DIR", "data/raw")
