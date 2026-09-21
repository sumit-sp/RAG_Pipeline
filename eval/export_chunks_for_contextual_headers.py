"""One-off export (not part of the permanent app): for every source document,
exports the full extracted text plus its own chunk list, so Claude Sonnet 5 can
generate a short contextual header per chunk externally (per DECISIONS.md,
"Groq restricted to generation only" — no LLM calls happen in this script).

This is the data for Phase 3's "contextual chunk headers" technique: a short
LLM-written blurb prepended to each chunk before embedding, situating it within
its source document (per Anthropic's "contextual retrieval" approach), meant to
help retrieval catch chunks whose own text alone is ambiguous out of context.

See eval/CONTEXTUAL_HEADERS_INSTRUCTIONS.md for the exact task given to Sonnet 5,
and app/pipelines/plain/contextual_headers.py for how the returned headers get
consumed at ingestion time.

Run with: python -m eval.export_chunks_for_contextual_headers
"""

import json
from pathlib import Path

from app.core import config
from app.pipelines.plain.ingestion import _chunker, _discover_source_files, _extract_text

OUTPUT_PATH = Path(__file__).parent / "chunks_for_contextual_headers.jsonl"


def main() -> None:
    raw_dir = Path(config.DATA_RAW_DIR)
    chunker = _chunker()
    print(f"Chunking strategy: {config.CHUNKING_STRATEGY} (size={config.CHUNK_SIZE_TOKENS}, overlap={config.CHUNK_OVERLAP_TOKENS})")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for path in _discover_source_files(raw_dir):
            source_doc = str(path.relative_to(raw_dir).as_posix())
            doc_type = path.relative_to(raw_dir).parts[0]
            full_text = _extract_text(path)
            chunks = chunker(full_text)

            record = {
                "source_doc": source_doc,
                "doc_type": doc_type,
                "full_text": full_text,
                "chunks": [
                    {"chunk_index": i, "text": chunk_text}
                    for i, chunk_text in enumerate(chunks)
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{source_doc}: {len(chunks)} chunks, {len(full_text)} chars")

    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
