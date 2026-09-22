"""One-off export (not part of the permanent app): runs the live pipeline
against all 41 golden_set.jsonl questions (not just the 15-question hard
subset from Step 8) and records retrieval + generation output for external
LLM-as-judge evaluation (Claude Sonnet 5, judged outside this repo, per
DECISIONS.md's Groq-restriction rule).

Current pipeline config: hybrid retrieval (dense + BM25) + fixed 500/50
chunking + Sonnet-5-generated contextual chunk headers baked into every chunk
before embedding (see EVALUATION_HISTORY.md Step 7).

No DeepEval, no Groq judge calls here — only retrieval (local) and generation
(Groq gpt-oss-20b, one call per question). Judging happens externally.

Run with: python -m eval.export_contextual_headers_full_for_external_judge
"""

import json
from pathlib import Path

from app.core import config
from app.pipelines.plain.generation import PlainGenerator
from app.pipelines.plain.retrieval import PlainRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
OUTPUT_PATH = Path(__file__).parent / "contextual_headers_full_outputs_for_external_judge.jsonl"


def main() -> None:
    golden_set = [
        json.loads(line)
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(
        f"config: RETRIEVAL_MODE={config.RETRIEVAL_MODE} "
        f"CHUNKING_STRATEGY={config.CHUNKING_STRATEGY} "
        f"USE_CONTEXTUAL_HEADERS={config.USE_CONTEXTUAL_HEADERS} (ingestion-time flag; "
        f"the live collection has headers baked in regardless) "
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
