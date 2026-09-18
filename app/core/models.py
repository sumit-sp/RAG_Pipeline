"""Shared dataclasses used by both the plain and langchain pipelines."""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A piece of a source document, ready to embed and index."""

    id: str
    text: str
    source_doc: str  # filename under data/raw/, e.g. "ai_act_2024_1689.html"
    doc_type: str  # "regulation" | "guidance" | "adjacent"
    chunk_index: int  # position of this chunk within its source document


@dataclass
class RetrievedContext:
    """A chunk returned by the retriever for a given query, with its relevance score."""

    chunk: Chunk
    score: float


@dataclass
class Answer:
    """The generator's response to a question."""

    text: str
    citations: list[str] = field(default_factory=list)  # distinct source_doc values used
