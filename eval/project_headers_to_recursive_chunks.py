"""Projects the existing fixed-chunking contextual headers
(eval/contextual_headers.jsonl) onto recursive chunking's different chunk
boundaries, without calling any LLM.

Why this is needed: contextual headers are LLM-generated content (Rule: any
LLM-judgment/content-generation task is exported for an external Sonnet-5
session, never re-run here -- see eval/CONTEXTUAL_HEADERS_INSTRUCTIONS.md).
Regenerating them properly for recursive chunking would mean repeating that
whole external process. This script instead reuses the already-approved
fixed-chunking headers via a deterministic, no-LLM mapping, using
eval/chunk_offsets.py's self-validating character-offset computation: every
recursive chunk is assigned the header of whichever fixed chunk it overlaps
with the most (by character count) -- an approximation (a recursive chunk
can span content two different fixed headers described), not a replacement
for real regeneration, but far better than either misaligned headers or no
headers at all.

Run with: python -m eval.project_headers_to_recursive_chunks
"""

import json
from pathlib import Path

from app.core import config
from app.pipelines.plain.contextual_headers import load_headers
from app.pipelines.plain.ingestion import _discover_source_files, _extract_text
from eval.chunk_offsets import ChunkOffset, chunks_with_offsets, overlap_chars

EVAL_DIR = Path(__file__).parent
OUTPUT_PATH = EVAL_DIR / "contextual_headers_recursive.jsonl"


def _best_overlap(rstart: int, rend: int, fixed: list[ChunkOffset]) -> int | None:
    best_idx, best_overlap_len = None, 0
    for i, (_, fstart, fend) in enumerate(fixed):
        overlap_len = overlap_chars(rstart, rend, fstart, fend)
        if overlap_len > best_overlap_len:
            best_idx, best_overlap_len = i, overlap_len
    return best_idx


def main() -> None:
    fixed_headers = load_headers(EVAL_DIR / "contextual_headers.jsonl")
    raw_dir = Path(config.DATA_RAW_DIR)

    projected = 0
    unmatched = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for path in _discover_source_files(raw_dir):
            source_doc = str(path.relative_to(raw_dir).as_posix())
            text = _extract_text(path)

            fixed_with_offsets = chunks_with_offsets(
                text, "fixed", config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
            )
            recursive_with_offsets = chunks_with_offsets(
                text, "recursive", config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
            )

            for i, (_, rstart, rend) in enumerate(recursive_with_offsets):
                best_idx = _best_overlap(rstart, rend, fixed_with_offsets)
                header = fixed_headers.get((source_doc, best_idx)) if best_idx is not None else None
                if header is None:
                    unmatched += 1
                    continue
                out.write(
                    json.dumps(
                        {
                            "source_doc": source_doc,
                            "chunk_index": i,
                            "header": header,
                            "projected_from_fixed_chunk_index": best_idx,
                        }
                    )
                    + "\n"
                )
                projected += 1

    print(f"Projected {projected} headers onto recursive chunks ({unmatched} unmatched) -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
