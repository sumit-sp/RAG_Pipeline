"""Phase 1 ingestion: naive parsing, fixed-size chunking, embed, store in Qdrant.

No per-document special-casing yet (that's Phase 4, only if eval proves it's
needed) — every .html file goes through BeautifulSoup, every .pdf through
PyMuPDF, and every resulting document goes through the same chunker.
"""

import uuid
from pathlib import Path

import pymupdf
from bs4 import BeautifulSoup
from qdrant_client.models import PointStruct

from app.core import config
from app.core.interfaces import Embedder
from app.core.models import Chunk
from app.pipelines.plain.chunking import chunk_text
from app.pipelines.plain.embedding import get_embedder
from app.pipelines.plain.vector_store import ensure_collection, get_qdrant_client

# Known navigation/index pages (link lists, not article content) — see
# data/raw/SOURCES.md. Excluded by this generic filename pattern rather than a
# per-document special case, so any future index-only page is skipped the same way.
_SKIP_PATTERN = "index"


def _extract_text(path: Path) -> str:
    if path.suffix == ".html":
        html = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
    if path.suffix == ".pdf":
        with pymupdf.open(path) as doc:
            return " ".join(page.get_text() for page in doc)
    raise ValueError(f"Unsupported file type: {path}")


def _discover_source_files(raw_dir: Path) -> list[Path]:
    files = [
        p
        for p in raw_dir.rglob("*")
        if p.suffix in (".html", ".pdf") and _SKIP_PATTERN not in p.name.lower()
    ]
    return sorted(files)


def _chunk_id(source_doc: str, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_doc}#{chunk_index}"))


class PlainIngestor:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or get_embedder()
        self.client = get_qdrant_client()

    def ingest(self, raw_dir: Path) -> int:
        source_files = _discover_source_files(raw_dir)
        all_chunks: list[Chunk] = []

        for path in source_files:
            text = _extract_text(path)
            source_doc = str(path.relative_to(raw_dir).as_posix())
            doc_type = path.relative_to(raw_dir).parts[0]
            for i, chunk_str in enumerate(chunk_text(text)):
                all_chunks.append(
                    Chunk(
                        id=_chunk_id(source_doc, i),
                        text=chunk_str,
                        source_doc=source_doc,
                        doc_type=doc_type,
                        chunk_index=i,
                    )
                )

        if not all_chunks:
            return 0

        vectors = self.embedder.embed_documents([c.text for c in all_chunks])
        ensure_collection(self.client, config.QDRANT_COLLECTION, vector_size=len(vectors[0]))

        points = [
            PointStruct(
                id=chunk.id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source_doc": chunk.source_doc,
                    "doc_type": chunk.doc_type,
                    "chunk_index": chunk.chunk_index,
                },
            )
            for chunk, vector in zip(all_chunks, vectors)
        ]
        self.client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
        return len(all_chunks)


if __name__ == "__main__":
    count = PlainIngestor().ingest(Path(config.DATA_RAW_DIR))
    print(f"Indexed {count} chunks from {config.DATA_RAW_DIR} into '{config.QDRANT_COLLECTION}'")
