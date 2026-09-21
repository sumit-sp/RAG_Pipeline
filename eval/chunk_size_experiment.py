"""One-off Phase 3 experiment script (not part of the permanent app): ingest the
corpus at a given CHUNK_SIZE_TOKENS/CHUNK_OVERLAP_TOKENS (read from env, set by the
caller before this process starts) into its own Qdrant collection, then report the
retrieval-hit rate against eval/golden_set.jsonl.

Deliberately Groq-free: no generation, no LLM-judged metrics. Only the
component-level, reference-based retrieval-hit check, so chunk-size candidates can
be screened before spending any Groq calls on the full eval suite.

Run once per candidate, e.g.:
  CHUNK_SIZE_TOKENS=800 CHUNK_OVERLAP_TOKENS=100 QDRANT_COLLECTION=ai_act_corpus_cs800 python eval/chunk_size_experiment.py
"""

import json
from pathlib import Path

from app.core import config
from app.pipelines.plain.ingestion import PlainIngestor
from app.pipelines.plain.retrieval import PlainRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"


def main() -> None:
    golden_set = [
        json.loads(line)
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(
        f"CHUNKING_STRATEGY={config.CHUNKING_STRATEGY} "
        f"CHUNK_SIZE_TOKENS={config.CHUNK_SIZE_TOKENS} "
        f"CHUNK_OVERLAP_TOKENS={config.CHUNK_OVERLAP_TOKENS} "
        f"collection={config.collection_name()}"
    )

    n_chunks = PlainIngestor().ingest(Path(config.DATA_RAW_DIR))
    print(f"Indexed {n_chunks} chunks")

    retriever = PlainRetriever()
    hits = 0
    misses = []
    for item in golden_set:
        contexts = retriever.retrieve(item["question"])
        retrieved_docs = {c.chunk.source_doc for c in contexts}
        if item["expected_source_doc"] in retrieved_docs:
            hits += 1
        else:
            misses.append((item["difficulty"], item["question"][:70]))

    print(f"Retrieval hit rate: {hits}/{len(golden_set)} ({100 * hits / len(golden_set):.0f}%)")
    for difficulty, question in misses:
        print(f"  MISS [{difficulty}] {question}")


if __name__ == "__main__":
    main()
