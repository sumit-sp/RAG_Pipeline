"""Migrates eval/qrels.jsonl (relevant chunks identified by chunk_index,
judged specifically against fixed chunking's boundaries) into
eval/qrels_spans.jsonl (relevant chunks identified by character span within
the source document), so relevance judgments work against ANY chunking
strategy's corpus, not just the one they were originally judged with.

Why chunk_index doesn't generalize: it's not a stable identifier, just "the
Nth chunk this particular chunker produced" -- a different chunker (e.g.
recursive) numbers the same document's content differently (954 chunks vs.
fixed's 837), so index equality across chunking strategies is coincidental,
not meaningful. A character span, by contrast, names an actual position in
the document's text, independent of how any chunker happens to slice it.

No new judging: this is a pure, deterministic re-representation of the
SAME already-approved relevance judgments (Step 14), using
eval/chunk_offsets.py's self-validating offset computation to convert each
judged fixed chunk_index into the (start_char, end_char) range it actually
covers. The granularity of "relevant" is still whatever a whole fixed
chunk covers (~500 tokens) -- this does not create a finer-grained ground
truth than what was actually judged, just a chunking-strategy-independent
representation of it.

Run with: python -m eval.migrate_qrels_to_char_spans
"""

import json
from pathlib import Path

from app.core import config
from app.pipelines.plain.ingestion import _discover_source_files, _extract_text
from eval.chunk_offsets import ChunkOffset, chunks_with_offsets

EVAL_DIR = Path(__file__).parent
QRELS_PATH = EVAL_DIR / "qrels.jsonl"
OUTPUT_PATH = EVAL_DIR / "qrels_spans.jsonl"


def _fixed_offsets_by_doc(raw_dir: Path) -> dict[str, list[ChunkOffset]]:
    offsets = {}
    for path in _discover_source_files(raw_dir):
        source_doc = str(path.relative_to(raw_dir).as_posix())
        text = _extract_text(path)
        offsets[source_doc] = chunks_with_offsets(
            text, "fixed", config.CHUNK_SIZE_TOKENS, config.CHUNK_OVERLAP_TOKENS
        )
    return offsets


def main() -> None:
    raw_dir = Path(config.DATA_RAW_DIR)
    offsets_by_doc = _fixed_offsets_by_doc(raw_dir)

    questions = 0
    spans_written = 0
    missing = []

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for line in QRELS_PATH.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            questions += 1

            relevant_spans = []
            for rc in rec["relevant_chunks"]:
                doc_offsets = offsets_by_doc.get(rc["source_doc"])
                if doc_offsets is None or rc["chunk_index"] >= len(doc_offsets):
                    missing.append((rec["question"][:60], rc["source_doc"], rc["chunk_index"]))
                    continue
                _, start, end = doc_offsets[rc["chunk_index"]]
                relevant_spans.append(
                    {
                        "source_doc": rc["source_doc"],
                        "start_char": start,
                        "end_char": end,
                        "reason": rc.get("reason", ""),
                    }
                )
                spans_written += 1

            out.write(json.dumps({"question": rec["question"], "relevant_spans": relevant_spans}) + "\n")

    print(f"Migrated {questions} questions, {spans_written} relevant spans -> {OUTPUT_PATH}")
    if missing:
        print(f"WARNING: {len(missing)} chunk references could not be resolved (out of range or unknown doc):")
        for q, doc, idx in missing:
            print(f"  {q!r} -- {doc}#{idx}")


if __name__ == "__main__":
    main()
