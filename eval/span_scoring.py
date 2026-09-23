"""Shared Precision/Recall scoring against eval/qrels_spans.jsonl (character
-span-based relevance judgments -- see eval/migrate_qrels_to_char_spans.py),
used by eval/compute_retrieval_metrics.py, dev_ui/eval_lookup.py, and
eval/query_decomposition_experiment.py. Kept in one place so all three stay
consistent rather than drifting apart as separate copies.

Works against ANY chunking strategy's corpus: a character span names an
actual position in the document text, independent of how any particular
chunker slices it. Caller must set config.CHUNKING_STRATEGY (and
CHUNK_SIZE_TOKENS/CHUNK_OVERLAP_TOKENS if non-default) to match whichever
corpus is actually being queried, so retrieved chunks' own spans are
computed with the right chunker.
"""

import json
from pathlib import Path

from app.core import config
from app.core.models import RetrievedContext
from app.pipelines.plain.ingestion import _extract_text
from eval.chunk_offsets import ChunkOffset, chunks_with_offsets, overlap_chars

QRELS_SPANS_PATH = Path(__file__).parent / "qrels_spans.jsonl"

# A retrieved chunk counts as relevant if it overlaps a relevant span by at
# least this fraction of the smaller of the two ranges -- a fraction rather
# than a fixed character count, so the threshold means the same thing
# regardless of a chunking strategy's typical chunk size. Filters out
# incidental edge-touching overlap (e.g. from fixed chunking's own
# deliberate inter-chunk overlap window) without requiring near-total
# containment.
MIN_OVERLAP_FRACTION = 0.2


def load_qrels_spans() -> dict[str, list[dict]]:
    if not QRELS_SPANS_PATH.exists():
        return {}
    qrels = {}
    for line in QRELS_SPANS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qrels[rec["question"]] = rec["relevant_spans"]
    return qrels


class ChunkSpanLookup:
    """Caches each (source document, chunking strategy)'s full chunk-offset
    list. Keyed on strategy too, not just source_doc -- a batch script only
    ever runs under one CHUNKING_STRATEGY for its whole process lifetime,
    but an interactive caller (the dev UI) can switch corpora mid-session,
    and a cache keyed on source_doc alone would then silently serve offsets
    computed under the wrong strategy for a previously-seen document."""

    def __init__(self, raw_dir: Path | None = None):
        self._raw_dir = raw_dir or Path(config.DATA_RAW_DIR)
        self._cache: dict[tuple[str, str], list[ChunkOffset]] = {}

    def span(self, source_doc: str, chunk_index: int) -> tuple[int, int]:
        key = (source_doc, config.CHUNKING_STRATEGY)
        if key not in self._cache:
            text = _extract_text(self._raw_dir / source_doc)
            self._cache[key] = chunks_with_offsets(
                text, config.CHUNKING_STRATEGY, config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
            )
        _, start, end = self._cache[key][chunk_index]
        return start, end


def is_relevant_match(source_doc: str, start: int, end: int, span: dict) -> bool:
    if span["source_doc"] != source_doc:
        return False
    overlap = overlap_chars(start, end, span["start_char"], span["end_char"])
    if overlap == 0:
        return False
    smaller_range = min(end - start, span["end_char"] - span["start_char"])
    return overlap >= MIN_OVERLAP_FRACTION * smaller_range


def precision_recall(
    contexts: list[RetrievedContext], relevant_spans: list[dict], lookup: ChunkSpanLookup
) -> tuple[float, float] | None:
    """None if there are no relevant spans for this question (undefined,
    not zero) -- same convention as the retired chunk_index-based version."""
    if not relevant_spans:
        return None

    chunk_spans = [(c.chunk.source_doc, *lookup.span(c.chunk.source_doc, c.chunk.chunk_index)) for c in contexts]

    hits = sum(
        1
        for source_doc, start, end in chunk_spans
        if any(is_relevant_match(source_doc, start, end, span) for span in relevant_spans)
    )
    covered = sum(
        1
        for span in relevant_spans
        if any(is_relevant_match(source_doc, start, end, span) for source_doc, start, end in chunk_spans)
    )

    precision = hits / len(contexts) if contexts else 0.0
    recall = covered / len(relevant_spans)
    return precision, recall
