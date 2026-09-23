"""Shared, self-validating character-offset computation for both chunking
strategies in app/pipelines/plain/chunking.py.

Every chunk (fixed or recursive) is an exact substring of its source
document's extracted text, so each chunk's (start_char, end_char) offset
within that document can be computed directly and deterministically, with
no LLM call. Kept as its own module -- separate from chunking.py -- so
one-off/offline scripts (header projection, qrels migration) can compute
offsets without touching core ingestion code, but shared here (not
duplicated per script) so there is exactly one place to fix if it ever
drifts from the real chunkers' behavior.

Self-validating: each public function recomputes chunk text from its own
offsets and asserts it exactly matches the real chunk_text()/
recursive_chunk_text() output before returning, so a caller can trust an
offset without separately re-checking it. If chunking.py's logic ever
changes, this fails loudly instead of silently producing wrong offsets.
"""

import tiktoken

from app.pipelines.plain.chunking import (
    _RECURSIVE_SEPARATORS,
    _token_len,
    chunk_text,
    recursive_chunk_text,
)

_encoding = tiktoken.get_encoding("cl100k_base")

# A single chunk's (text, start_char, end_char) within its source document.
ChunkOffset = tuple[str, int, int]


def fixed_chunks_with_offsets(text: str, size: int, overlap: int) -> list[ChunkOffset]:
    """Mirrors chunking.chunk_text, also returning each chunk's (start, end)
    character offset within `text`."""
    tokens = _encoding.encode(text)
    if not tokens:
        return []
    step = size - overlap
    results: list[ChunkOffset] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + size]
        chunk_str = _encoding.decode(window)
        char_start = len(_encoding.decode(tokens[:start])) if start else 0
        results.append((chunk_str, char_start, char_start + len(chunk_str)))
        if start + size >= len(tokens):
            break
        start += step

    assert [c for c, _, _ in results] == chunk_text(text, size=size, overlap=overlap), (
        "fixed_chunks_with_offsets drifted from the real chunk_text()"
    )
    return results


def _split_into_pieces_with_offsets(
    text: str, offset: int, size: int, separators: list[str]
) -> list[ChunkOffset]:
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

    pieces: list[ChunkOffset] = []
    cursor = offset
    for part in parts:
        pieces.extend(_split_into_pieces_with_offsets(part, cursor, size, rest))
        cursor += len(part)
    return pieces


def _merge_pieces_with_offsets(pieces: list[ChunkOffset], size: int, overlap: int) -> list[ChunkOffset]:
    """Mirrors chunking._merge_pieces, carrying offsets through so each output
    chunk's (start, end) character range in the original document is known."""
    chunks: list[ChunkOffset] = []
    current: list[ChunkOffset] = []
    current_len = 0

    def _flush(pieces_group: list[ChunkOffset]) -> ChunkOffset:
        text = "".join(p for p, _, _ in pieces_group)
        return text, pieces_group[0][1], pieces_group[-1][2]

    for piece in pieces:
        piece_len = _token_len(piece[0])
        if current and current_len + piece_len > size:
            chunks.append(_flush(current))
            overlap_pieces: list[ChunkOffset] = []
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


def recursive_chunks_with_offsets(text: str, size: int, overlap: int) -> list[ChunkOffset]:
    """Mirrors chunking.recursive_chunk_text, also returning each chunk's
    (start, end) character offset within `text`."""
    if not text.strip():
        return []
    pieces = _split_into_pieces_with_offsets(text, 0, size, list(_RECURSIVE_SEPARATORS))
    results = _merge_pieces_with_offsets(pieces, size, overlap)

    assert [c for c, _, _ in results] == recursive_chunk_text(text, size=size, overlap=overlap), (
        "recursive_chunks_with_offsets drifted from the real recursive_chunk_text()"
    )
    return results


def chunks_with_offsets(text: str, strategy: str, size: int, overlap: int) -> list[ChunkOffset]:
    """Dispatches to the offset-computing mirror of whichever chunker
    `strategy` names -- same dispatch rule as ingestion.py's `_chunker()`."""
    if strategy == "recursive":
        return recursive_chunks_with_offsets(text, size, overlap)
    return fixed_chunks_with_offsets(text, size, overlap)


def overlap_chars(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    """Character-length overlap between two [start, end) ranges (0 if none)."""
    return max(0, min(a_end, b_end) - max(a_start, b_start))
