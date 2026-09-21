"""Chunking strategies. `chunk_text` is Phase 1's fixed-size token window — no
cleverness, the dumbest complete pipeline. `recursive_chunk_text` is a Phase 3
chunk-size-tuning candidate: it prefers to split on natural text boundaries
(paragraph, line, sentence, word) before falling back to a hard token cut, so
chunks are less likely to land mid-sentence.

Kept as their own module (rather than inlined in ingestion.py) because chunking is
the one piece of Phase 1 logic worth unit-testing on its own in Phase 5.
"""

import tiktoken

from app.core import config

_encoding = tiktoken.get_encoding("cl100k_base")

# Tried in order: paragraph break, line break, sentence end, word boundary. Falling
# back to "" means splitting character-by-character, for the rare piece (e.g. a
# run-on legal sentence) still too long after all of the above.
_RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _token_len(text: str) -> int:
    return len(_encoding.encode(text))


def chunk_text(
    text: str,
    size: int = config.CHUNK_SIZE_TOKENS,
    overlap: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text into overlapping windows of `size` tokens, sliding by `size - overlap`."""
    tokens = _encoding.encode(text)
    if not tokens:
        return []

    step = size - overlap
    chunks = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + size]
        chunks.append(_encoding.decode(window))
        if start + size >= len(tokens):
            break
        start += step
    return chunks


def _split_into_pieces(text: str, size: int, separators: list[str]) -> list[str]:
    """Recursively break text into pieces no larger than `size` tokens, preferring
    the earliest separator in `separators` that actually splits the text. Each
    piece keeps its trailing separator (except the last piece), so joining every
    piece back together with "".join() exactly reconstructs the original text —
    that's what lets _merge_pieces re-pack them with plain "".join()."""
    if _token_len(text) <= size or not separators:
        return [text] if text else []

    separator, *rest = separators
    if separator == "":
        parts = list(text)
    else:
        split = text.split(separator)
        # re-attach the separator to every piece except the last, so no text is lost
        parts = [p + separator for p in split[:-1]] + split[-1:]
    parts = [p for p in parts if p]

    if len(parts) <= 1:
        return _split_into_pieces(text, size, rest)

    pieces = []
    for part in parts:
        pieces.extend(_split_into_pieces(part, size, rest))
    return pieces


def _merge_pieces(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily pack small pieces into chunks up to `size` tokens, carrying the
    trailing ~`overlap` tokens of one chunk into the start of the next."""
    chunks = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        piece_len = _token_len(piece)
        if current and current_len + piece_len > size:
            chunks.append("".join(current))

            overlap_pieces: list[str] = []
            overlap_len = 0
            for p in reversed(current):
                p_len = _token_len(p)
                if overlap_len + p_len > overlap:
                    break
                overlap_pieces.insert(0, p)
                overlap_len += p_len
            current, current_len = overlap_pieces, overlap_len

        current.append(piece)
        current_len += piece_len

    if current:
        chunks.append("".join(current))
    return chunks


def recursive_chunk_text(
    text: str,
    size: int = config.CHUNK_SIZE_TOKENS,
    overlap: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Split text on natural boundaries (paragraph/line/sentence/word) where
    possible, only cutting mid-word as a last resort, then pack the resulting
    pieces into ~`size`-token chunks with ~`overlap` tokens of carry-over."""
    if not text.strip():
        return []
    pieces = _split_into_pieces(text, size, _RECURSIVE_SEPARATORS)
    return _merge_pieces(pieces, size, overlap)
