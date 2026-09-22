"""One-off export (not part of the permanent app): re-runs the live pipeline
against exactly the golden-set questions the Step 12 cross-reference boost
(app/pipelines/plain/retrieval.py) actually fires on, and records
retrieval + generation output for external LLM-as-judge evaluation (Claude
Sonnet 5, judged outside this repo, per DECISIONS.md's Groq-restriction rule).

Why this subset and not all 41: the boost is the only thing that changed
since Step 10's full-set judging. The other ~26 questions get byte-identical
retrieval to Step 10 (same collection, same retrieval logic minus the boost
firing), so re-judging them would just reproduce Step 10's scores at the
cost of more judge effort -- exactly the kind of unnecessary Tier-2 spend the
spec's evaluation-cost cascade says to avoid. Judging only the questions the
change actually touches gives a direct before/after comparison against
Step 10's existing scores for the same questions.

No DeepEval, no Groq judge calls here -- only retrieval (local) and
generation (Groq gpt-oss-20b, one call per question). Judging happens
externally.

Run with: python -m eval.export_boosted_subset_for_external_judge
"""

import json
from pathlib import Path

from app.core import config
from app.pipelines.plain.generation import PlainGenerator
from app.pipelines.plain.retrieval import PlainRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
OUTPUT_PATH = Path(__file__).parent / "boosted_subset_outputs_for_external_judge.jsonl"


def main() -> None:
    golden_set = [
        json.loads(line)
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    print(
        f"config: RETRIEVAL_MODE={config.RETRIEVAL_MODE} "
        f"USE_CROSS_REFERENCE_BOOST={config.USE_CROSS_REFERENCE_BOOST} "
        f"collection={config.collection_name()}"
    )

    retriever = PlainRetriever()
    generator = PlainGenerator()

    records = []
    for item in golden_set:
        contexts = retriever.retrieve(item["question"])
        if len(contexts) <= config.RETRIEVAL_TOP_K:
            continue  # boost didn't fire for this question -- unchanged since Step 10
        boosted_docs = sorted(
            {c.chunk.source_doc for c in contexts[config.RETRIEVAL_TOP_K :]}
        )
        answer = generator.generate(item["question"], contexts)
        records.append(
            {
                "question": item["question"],
                "difficulty": item["difficulty"],
                "expected_answer": item["expected_answer"],
                "expected_source_doc": item["expected_source_doc"],
                "expected_source_section": item["expected_source_section"],
                "boosted_in_documents": boosted_docs,
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
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, record in enumerate(records, start=1):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(records)}] {record['question'][:70]}")

    print(f"Wrote {len(records)} boosted-subset records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
