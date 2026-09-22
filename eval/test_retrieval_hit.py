"""Component-level retriever check, deterministic and reference-based — no LLM
calls at all (no generation, no judge). Safe to run automatically in CI: it costs
nothing, needs no API key, and gives a real regression signal on retrieval quality
whenever chunking, embeddings, or retrieval logic changes.

Gated on an aggregate hit-rate threshold rather than per-question assertions,
since several individual misses are already known and documented (see
EVALUATION_HISTORY.md) — this test should catch new regressions below the
current baseline, not stay permanently red over pre-existing, tracked gaps.

Full LLM-judged quality evaluation (faithfulness, answer correctness, etc.) is a
separate, manual process — see eval/export_for_external_judge.py and
DECISIONS.md ("Groq restricted to generation only").
"""

import json
from pathlib import Path

from app.pipelines.plain.retrieval import PlainRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"

# Current baseline is 40/41 (97.6%) — contextual chunk headers (Step 7) + the
# corpus-side cross-reference boost (Step 12, generalizes Step 11's GDPR-only
# keyword version) — see EVALUATION_HISTORY.md. Set a few points below that so
# normal run-to-run determinism doesn't false-positive, while still catching a
# real regression.
MIN_HIT_RATE = 0.93


def _load_golden_set() -> list[dict]:
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_retrieval_hit_rate():
    golden_set = _load_golden_set()
    retriever = PlainRetriever()

    misses = []
    for item in golden_set:
        contexts = retriever.retrieve(item["question"])
        retrieved_docs = {c.chunk.source_doc for c in contexts}
        if item["expected_source_doc"] not in retrieved_docs:
            misses.append(item["question"])

    hit_rate = 1 - len(misses) / len(golden_set)
    print(f"\nRetrieval hit rate: {hit_rate:.1%} ({len(golden_set) - len(misses)}/{len(golden_set)})")
    for question in misses:
        print(f"  MISS: {question[:80]}")

    assert hit_rate >= MIN_HIT_RATE, (
        f"Retrieval hit rate {hit_rate:.1%} dropped below the {MIN_HIT_RATE:.0%} "
        f"floor — {len(misses)} misses: {misses}"
    )
