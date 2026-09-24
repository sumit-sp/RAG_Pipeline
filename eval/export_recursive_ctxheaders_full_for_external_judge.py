"""One-off export (not part of the permanent app): runs the live pipeline
against all 41 golden_set.jsonl questions, against the recursive-chunking +
projected-headers corpus (ai_act_corpus_recursive_ctxheaders_hybrid), and
records retrieval + generation output for external LLM-as-judge evaluation
(Claude Sonnet 5, judged outside this repo, per DECISIONS.md's
Groq-restriction rule).

Why this export exists: Step 20 found this corpus beats current production
on deterministic retrieval Precision/Recall, but that's a retrieval-only
metric -- current production earned its spot through a full external
Sonnet-5 judging pass on actual generated answers (Steps 6/8/10), which
this corpus has never been through (only one question was manually
inspected, Step 18). Before considering this for production, get the same
judging rigor on actual answer quality, not just retrieved-chunk quality.

Pipeline config for this export: hybrid retrieval (dense + BM25) +
recursive chunking + contextual chunk headers PROJECTED from the original
fixed-chunking headers (a character-offset-overlap approximation, not an
independent Sonnet-5 generation for this chunking -- see
EVALUATION_HISTORY.md Steps 17/20 for how these were produced).

No DeepEval, no Groq judge calls here -- only retrieval (local) and
generation (Groq, one call per question). Judging happens externally.

Run with: python -m eval.export_recursive_ctxheaders_full_for_external_judge
"""

import json
from pathlib import Path

from app.core import config
from app.pipelines.plain.generation import PlainGenerator
from app.pipelines.plain.retrieval import PlainRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
OUTPUT_PATH = Path(__file__).parent / "recursive_ctxheaders_full_outputs_for_external_judge.jsonl"


def main() -> None:
    config.QDRANT_COLLECTION = "ai_act_corpus_recursive_ctxheaders"
    config.CHUNKING_STRATEGY = "recursive"
    config.USE_CONTEXTUAL_HEADERS = True
    config.CONTEXTUAL_HEADERS_PATH = "eval/contextual_headers_recursive.jsonl"

    golden_set = [
        json.loads(line)
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(
        f"config: RETRIEVAL_MODE={config.RETRIEVAL_MODE} "
        f"CHUNKING_STRATEGY={config.CHUNKING_STRATEGY} "
        f"collection={config.collection_name()}"
    )

    retriever = PlainRetriever()
    generator = PlainGenerator()

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, item in enumerate(golden_set, start=1):
            contexts = retriever.retrieve(item["question"])
            answer = generator.generate(item["question"], contexts)

            record = {
                "question": item["question"],
                "difficulty": item["difficulty"],
                "expected_answer": item["expected_answer"],
                "expected_source_doc": item["expected_source_doc"],
                "expected_source_section": item["expected_source_section"],
                "generated_answer": answer.text,
                "citations": answer.citations,
                "retrieved_chunks": [
                    {
                        "rank": rank,
                        "source_doc": c.chunk.source_doc,
                        "chunk_index": c.chunk.chunk_index,
                        "score": c.score,
                        "text": c.chunk.text,
                    }
                    for rank, c in enumerate(contexts, start=1)
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(golden_set)}] {item['question'][:70]}")

    print(f"Wrote {len(golden_set)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
