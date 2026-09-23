"""Projects the existing fixed-chunking contextual headers
(eval/contextual_headers.jsonl) onto recursive chunking's different chunk
boundaries, without calling any LLM.

Why this is needed: contextual headers are LLM-generated content (Rule: any
LLM-judgment/content-generation task is exported for an external Sonnet-5
session, never re-run here -- see eval/CONTEXTUAL_HEADERS_INSTRUCTIONS.md).
Regenerating them properly for recursive chunking would mean repeating that
whole external process. This script instead reuses the already-approved
fixed-chunking headers via a deterministic, no-LLM mapping: every chunk
(fixed or recursive) is an exact substring of the same source document, so
each chunk's (start_char, end_char) offset in that document can be computed
directly. Every recursive chunk is then assigned the header of whichever
fixed chunk it overlaps with the most (by character count) -- an
approximation (a recursive chunk can span content two different fixed
headers described), not a replacement for real regeneration, but far better
than either misaligned headers or no headers at all.

Self-validating: the offset-computing mirror functions below are a separate
implementation from chunking.py's real chunk_text/recursive_chunk_text (to
avoid modifying core ingestion code for a one-off migration script), so this
script re-derives chunk text from its own offsets and asserts it exactly
matches the real chunkers' actual output for every document before trusting
any offset. If chunking.py's logic ever changes, this assertion fails loudly
instead of silently producing wrong offsets.

Run with: python -m eval.project_headers_to_recursive_chunks
"""

import json
from pathlib import Path

import tiktoken

from app.core import config
from app.pipelines.plain.chunking import _RECURSIVE_SEPARATORS, _token_len, chunk_text, recursive_chunk_text
from app.pipelines.plain.contextual_headers import load_headers
from app.pipelines.plain.ingestion import _discover_source_files, _extract_text

_encoding = tiktoken.get_encoding("cl100k_base")

EVAL_DIR = Path(__file__).parent
OUTPUT_PATH = EVAL_DIR / "contextual_headers_recursive.jsonl"


def _fixed_chunks_with_offsets(text: str, size: int, overlap: int) -> list[tuple[str, int, int]]:
    """Mirrors chunking.chunk_text, but also returns each chunk's (start, end)
    character offset within `text`."""
    tokens = _encoding.encode(text)
    if not tokens:
        return []
    step = size - overlap
    results = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + size]
        chunk_str = _encoding.decode(window)
        char_start = len(_encoding.decode(tokens[:start])) if start else 0
        results.append((chunk_str, char_start, char_start + len(chunk_str)))
        if start + size >= len(tokens):
            break
        start += step
    return results


def _split_into_pieces_with_offsets(
    text: str, offset: int, size: int, separators: list[str]
) -> list[tuple[str, int, int]]:
    """Mirrors chunking._split_into_pieces, tracking each piece's (start, end)
    offset within the ORIGINAL document `offset` refers into (not `text`,
    which is itself a sub-piece during recursion)."""
    if _token_len(text) <= size or not separators:
        return [(text, offset, offset + len(text))] if text else []

    separator, *rest = separators
    if separator == "":
        parts = list(text)
    else:
        split = text.split(separator)
        parts = [p + separator for p in split[:-1]] + split[-1:]
    parts = [p for p in parts if p]

    if len(parts) <= 1:
        return _split_into_pieces_with_offsets(text, offset, size, rest)

    pieces = []
    cursor = offset
    for part in parts:
        pieces.extend(_split_into_pieces_with_offsets(part, cursor, size, rest))
        cursor += len(part)
    return pieces


def _merge_pieces_with_offsets(
    pieces: list[tuple[str, int, int]], size: int, overlap: int
) -> list[tuple[str, int, int]]:
    """Mirrors chunking._merge_pieces, carrying offsets through so each output
    chunk's (start, end) character range in the original document is known."""
    chunks: list[tuple[str, int, int]] = []
    current: list[tuple[str, int, int]] = []
    current_len = 0

    def _flush(pieces_group: list[tuple[str, int, int]]) -> tuple[str, int, int]:
        text = "".join(p for p, _, _ in pieces_group)
        return text, pieces_group[0][1], pieces_group[-1][2]

    for piece in pieces:
        piece_len = _token_len(piece[0])
        if current and current_len + piece_len > size:
            chunks.append(_flush(current))
            overlap_pieces: list[tuple[str, int, int]] = []
            overlap_len = 0
            for p in reversed(current):
                p_len = _token_len(p[0])
                if overlap_len + p_len > overlap:
                    break
                overlap_pieces.insert(0, p)
                overlap_len += p_len
            current, current_len = overlap_pieces, overlap_len

        current.append(piece)
        current_len += piece_len

    if current:
        chunks.append(_flush(current))
    return chunks


def _recursive_chunks_with_offsets(text: str, size: int, overlap: int) -> list[tuple[str, int, int]]:
    if not text.strip():
        return []
    pieces = _split_into_pieces_with_offsets(text, 0, size, list(_RECURSIVE_SEPARATORS))
    return _merge_pieces_with_offsets(pieces, size, overlap)


def _best_overlap(rstart: int, rend: int, fixed: list[tuple[str, int, int]]) -> int | None:
    best_idx, best_overlap_len = None, 0
    for i, (_, fstart, fend) in enumerate(fixed):
        overlap_len = max(0, min(rend, fend) - max(rstart, fstart))
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

            fixed_with_offsets = _fixed_chunks_with_offsets(
                text, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
            )
            assert [c for c, _, _ in fixed_with_offsets] == chunk_text(text), (
                f"offset mirror drifted from the real chunk_text() for {source_doc}"
            )

            recursive_with_offsets = _recursive_chunks_with_offsets(
                text, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
            )
            assert [c for c, _, _ in recursive_with_offsets] == recursive_chunk_text(text), (
                f"offset mirror drifted from the real recursive_chunk_text() for {source_doc}"
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
