"""Computes real, deterministic Precision@k and Recall@k for the live
retrieval pipeline against eval/qrels.jsonl (Claude Sonnet 5's relevance
judgments over a pooled candidate set — see eval/QRELS_JUDGE_INSTRUCTIONS.md
for the pooling methodology and its recall-is-pool-bounded caveat).

No LLM calls -- once the qrels file exists, this is pure set arithmetic, and
can be re-run for free against any future retrieval config change.

Run with: python -m eval.compute_retrieval_metrics
"""

import json
import statistics
from pathlib import Path

from app.pipelines.plain.retrieval import PlainRetriever

EVAL_DIR = Path(__file__).parent
QRELS_PATH = EVAL_DIR / "qrels.jsonl"


def _load_qrels() -> dict[str, set[tuple[str, int]]]:
    qrels = {}
    for line in QRELS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qrels[rec["question"]] = {
            (rc["source_doc"], rc["chunk_index"]) for rc in rec["relevant_chunks"]
        }
    return qrels


def main() -> None:
    qrels = _load_qrels()
    retriever = PlainRetriever()

    precisions, recalls, f1s = [], [], []
    zero_relevant_questions = []

    print(f"{'Precision':>10} {'Recall':>8} {'Retrieved':>10} {'Relevant':>9}  Question")
    for question, relevant in qrels.items():
        contexts = retriever.retrieve(question)
        retrieved = {(c.chunk.source_doc, c.chunk.chunk_index) for c in contexts}
        hits = retrieved & relevant

        if not relevant:
            zero_relevant_questions.append(question)
            continue  # precision/recall undefined (no relevant chunks in the pool at all)

        precision = len(hits) / len(retrieved) if retrieved else 0.0
        recall = len(hits) / len(relevant)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(f"{precision:>10.2f} {recall:>8.2f} {len(retrieved):>10} {len(relevant):>9}  {question[:70]}")

    print(f"\n{'='*100}")
    print(f"N = {len(precisions)} questions (excluding {len(zero_relevant_questions)} with zero relevant chunks in their pool)")
    print(f"Mean Precision: {statistics.mean(precisions):.3f}")
    print(f"Mean Recall:    {statistics.mean(recalls):.3f}  (bounded by the candidate pool, not the full corpus -- see QRELS_JUDGE_INSTRUCTIONS.md)")
    print(f"Mean F1:        {statistics.mean(f1s):.3f}")
    if zero_relevant_questions:
        print("\nQuestions with zero relevant chunks in their candidate pool (excluded above):")
        for q in zero_relevant_questions:
            print(f"  - {q[:90]}")


if __name__ == "__main__":
    main()
