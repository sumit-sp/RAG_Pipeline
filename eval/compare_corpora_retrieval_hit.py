"""Step 18 systematic diagnostic, part 1: is the recursive-chunking citation
regression found manually on one question isolated, or systematic?

Runs the same Groq-free, deterministic retrieval-hit check as
eval/test_retrieval_hit.py against all 41 golden-set questions, once per
named corpus (Step 17), and diffs the pass/fail sets pairwise. No LLM calls,
no re-ingestion -- just querying the 4 collections that already exist.

Run with: python -m eval.compare_corpora_retrieval_hit
"""

import json
from pathlib import Path

from app.core import config
from app.pipelines.plain.retrieval import PlainRetriever

EVAL_DIR = Path(__file__).parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.jsonl"

# base collection name -> (label, actually built with headers?)
CORPORA = {
    "ai_act_corpus": "Production (fixed + headers)",
    "ai_act_corpus_fixed": "Fixed only (no headers)",
    "ai_act_corpus_recursive": "Recursive only (no headers)",
    "ai_act_corpus_recursive_ctxheaders": "Recursive + headers (projected)",
}


def _load_golden_set() -> list[dict]:
    return [json.loads(line) for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    golden_set = _load_golden_set()
    retriever = PlainRetriever()  # embedder/sparse embedder/client are shared across corpora

    # {base_name: {question: hit_bool}}
    results: dict[str, dict[str, bool]] = {}

    for base_name, label in CORPORA.items():
        config.QDRANT_COLLECTION = base_name
        hits = {}
        for item in golden_set:
            contexts = retriever.retrieve(item["question"])
            retrieved_docs = {c.chunk.source_doc for c in contexts}
            hits[item["question"]] = item["expected_source_doc"] in retrieved_docs
        results[base_name] = hits
        rate = sum(hits.values())
        print(f"{label:45s} ({base_name}): {rate}/{len(golden_set)} ({rate / len(golden_set):.1%})")

    print()
    print("=" * 100)
    print("Pairwise diffs (base vs. comparison) -- questions where the pass/fail flipped")
    print("=" * 100)

    pairs = [
        ("ai_act_corpus_fixed", "ai_act_corpus_recursive", "Fixed(no headers) -> Recursive(no headers)"),
        ("ai_act_corpus", "ai_act_corpus_recursive_ctxheaders", "Production(fixed+headers) -> Recursive+headers"),
        ("ai_act_corpus_recursive", "ai_act_corpus_recursive_ctxheaders", "Recursive(no headers) -> Recursive+headers"),
    ]

    difficulty_by_question = {item["question"]: item.get("difficulty") for item in golden_set}

    for base, other, label in pairs:
        print(f"\n--- {label} ---")
        regressions = [q for q in results[base] if results[base][q] and not results[other][q]]
        improvements = [q for q in results[base] if not results[base][q] and results[other][q]]
        print(f"Regressions (passed under '{base}', failed under '{other}'): {len(regressions)}")
        for q in regressions:
            print(f"  [{difficulty_by_question[q]}] {q[:100]}")
        print(f"Improvements (failed under '{base}', passed under '{other}'): {len(improvements)}")
        for q in improvements:
            print(f"  [{difficulty_by_question[q]}] {q[:100]}")


if __name__ == "__main__":
    main()
