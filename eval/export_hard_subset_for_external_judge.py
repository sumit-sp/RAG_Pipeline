"""One-off export (not part of the permanent app): runs the live pipeline
against a 15-question "hard subset" of golden_set.jsonl and records retrieval
+ generation output for external LLM-as-judge evaluation (Claude Sonnet 5,
judged outside this repo, per DECISIONS.md's Groq-restriction rule).

The hard subset = every cross-reference/multi-hop/temporal-conflict question
(13) plus the 2 single-hop questions that miss the Groq-free retrieval-hit
check against the current (contextual-headers-enabled) index — 15 total. This
targets judging effort at the questions most likely to actually fail, rather
than spending 41 judge-calls' worth of the user's time on mostly-easy
single-hop questions.

No DeepEval, no Groq judge calls here — only retrieval (local) and generation
(Groq gpt-oss-20b, one call per question). Judging happens externally.

Run with the current default pipeline config already active (hybrid retrieval
+ contextual headers baked into the ingested collection):
  python -m eval.export_hard_subset_for_external_judge
"""

import json
from pathlib import Path

from app.core import config
from app.pipelines.plain.generation import PlainGenerator
from app.pipelines.plain.retrieval import PlainRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
OUTPUT_PATH = Path(__file__).parent / "hard_subset_outputs_for_external_judge.jsonl"

_HARD_DIFFICULTIES = {"cross-reference", "multi-hop", "temporal-conflict"}

# The 2 single-hop questions that miss the Groq-free retrieval-hit check
# against the contextual-headers-enabled index (see EVALUATION_HISTORY.md
# Step 7) — hard in practice even though labeled single-hop by structure.
_EXTRA_SINGLE_HOP_MISSES = {
    "What was the original application date for the AI Act's high-risk system requirements (Chapter III), before the Digital Omnibus amendment?",
    "Under Article 53 of the AI Act, are providers of open-source general-purpose AI models exempt from all obligations in Article 53(1)?",
}


def _select_hard_subset(golden_set: list[dict]) -> list[dict]:
    return [
        item
        for item in golden_set
        if item["difficulty"] in _HARD_DIFFICULTIES or item["question"] in _EXTRA_SINGLE_HOP_MISSES
    ]


def main() -> None:
    golden_set = [
        json.loads(line)
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subset = _select_hard_subset(golden_set)

    print(
        f"config: RETRIEVAL_MODE={config.RETRIEVAL_MODE} "
        f"CHUNKING_STRATEGY={config.CHUNKING_STRATEGY} "
        f"USE_CONTEXTUAL_HEADERS={config.USE_CONTEXTUAL_HEADERS} "
        f"collection={config.collection_name()}"
    )
    print(f"Hard subset: {len(subset)} questions (expected 15)")

    retriever = PlainRetriever()
    generator = PlainGenerator()

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, item in enumerate(subset, start=1):
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
            print(f"[{i}/{len(subset)}] {item['question'][:70]}")

    print(f"Wrote {len(subset)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
