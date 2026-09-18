"""Fixed-size token chunking. No cleverness — Phase 1 is the dumbest complete pipeline.

Kept as its own module (rather than inlined in ingestion.py) because it's the one
piece of Phase 1 logic worth unit-testing on its own in Phase 5.
"""

import tiktoken

from app.core import config

_encoding = tiktoken.get_encoding("cl100k_base")


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
