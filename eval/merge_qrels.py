"""Merges the 7 judged batches (eval/qrels_batches/batch_NN_judged.jsonl,
Claude Sonnet 5's relevance judgments) into a single canonical
eval/qrels.jsonl, validating along the way:

1. Every one of the 41 golden-set questions appears exactly once across all
   batches (no question dropped, none duplicated).
2. Every judged-relevant chunk actually exists in that question's original
   candidate pool (eval/qrels_candidates.jsonl) -- catches a hallucinated or
   mistyped (source_doc, chunk_index) reference before it silently corrupts
   downstream precision/recall numbers.

Run with: python -m eval.merge_qrels
"""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent
BATCHES_DIR = EVAL_DIR / "qrels_batches"
CANDIDATES_PATH = EVAL_DIR / "qrels_candidates.jsonl"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.jsonl"
OUTPUT_PATH = EVAL_DIR / "qrels.jsonl"


def main() -> None:
    golden_questions = {
        json.loads(line)["question"]
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    candidate_pools = {}
    for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        candidate_pools[rec["question"]] = {
            (c["source_doc"], c["chunk_index"]) for c in rec["candidates"]
        }

    judged_batches = sorted(BATCHES_DIR.glob("batch_*_judged.jsonl"))
    print(f"Found {len(judged_batches)} judged batch files")

    merged: dict[str, dict] = {}
    errors: list[str] = []

    for path in judged_batches:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            question = rec["question"]

            if question in merged:
                errors.append(f"DUPLICATE question across batches: {question[:70]!r}")
            if question not in golden_questions:
                errors.append(f"UNKNOWN question not in golden_set.jsonl: {question[:70]!r}")

            pool = candidate_pools.get(question, set())
            valid_relevant = []
            for rc in rec.get("relevant_chunks", []):
                key = (rc["source_doc"], rc["chunk_index"])
                if key not in pool:
                    errors.append(
                        f"HALLUCINATED chunk {key} for question {question[:60]!r} "
                        f"(not in original candidate pool, from {path.name})"
                    )
                    continue
                valid_relevant.append(rc)

            merged[question] = {"question": question, "relevant_chunks": valid_relevant}

    missing = golden_questions - merged.keys()
    for q in missing:
        errors.append(f"MISSING question, not judged in any batch: {q[:70]!r}")

    if errors:
        print(f"\n{len(errors)} problem(s) found:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nNo problems found: all 41 questions present exactly once, "
              "every judged chunk verified against its candidate pool.")

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for question in sorted(merged):
            f.write(json.dumps(merged[question], ensure_ascii=False) + "\n")

    counts = [len(v["relevant_chunks"]) for v in merged.values()]
    zero_count = sum(1 for c in counts if c == 0)
    print(
        f"\nWrote {len(merged)} questions to {OUTPUT_PATH}\n"
        f"Relevant chunks per question: min={min(counts)} max={max(counts)} "
        f"mean={sum(counts) / len(counts):.2f} total={sum(counts)}\n"
        f"Questions with zero relevant chunks in their pool: {zero_count}"
    )


if __name__ == "__main__":
    main()
