"""Contracts shared by the plain/ and langchain/ pipeline implementations.

Defined once, early, per the project spec — both backends satisfy these same
Protocols so the API layer can swap between them via the PIPELINE_BACKEND env var
without caring which one it's talking to.
"""

from pathlib import Path
from typing import Protocol

from app.core.models import Answer, Chunk, RetrievedContext


class Embedder(Protocol):
    """Turns text into vectors. Swappable independently of the pipeline backend
    (local model vs. a hosted API) via the EMBEDDING_BACKEND env var."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class Ingestor(Protocol):
    """Parses source documents, chunks them, embeds the chunks, and stores them
    in the vector store. One offline batch job, not a per-request operation."""

    def ingest(self, raw_dir: Path) -> int:
        """Index every document under raw_dir. Returns the number of chunks indexed."""
        ...


class Retriever(Protocol):
    """Finds the chunks most relevant to a question."""

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedContext]: ...


class Generator(Protocol):
    """Turns a question + retrieved context into a cited answer."""

    def generate(self, question: str, contexts: list[RetrievedContext]) -> Answer: ...
