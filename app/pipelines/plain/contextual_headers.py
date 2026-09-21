"""Loads externally-generated contextual chunk headers (see
eval/CONTEXTUAL_HEADERS_INSTRUCTIONS.md and DECISIONS.md — "Groq restricted to
generation only": these headers are written by Claude Sonnet 5 outside this repo,
never by Groq) and prepends them to chunk text before embedding, per Anthropic's
"contextual retrieval" technique.
"""

import json
from pathlib import Path

_SEPARATOR = "\n\n"


def load_headers(path: Path) -> dict[tuple[str, int], str]:
    if not path.exists():
        return {}
    headers = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            headers[(record["source_doc"], record["chunk_index"])] = record["header"]
    return headers


def apply_header(text: str, header: str | None) -> str:
    if not header:
        return text
    return f"{header}{_SEPARATOR}{text}"
