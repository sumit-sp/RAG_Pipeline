"""Computes real, deterministic Precision@k and Recall@k for the live
retrieval pipeline against eval/qrels_spans.jsonl (character-span-based
relevance judgments), via eval/span_scoring.py -- works against ANY
chunking strategy's corpus, unlike the retired chunk_index-based
eval/qrels.jsonl, which only applied to the exact fixed-chunking corpus it
was originally judged against.

Set CHUNKING_STRATEGY (and QDRANT_COLLECTION/QDRANT_URL, as usual) to match
whichever corpus PlainRetriever is actually pointed at.

No LLM calls -- once qrels_spans.jsonl exists, this is pure text-offset
arithmetic, re-runnable for free against any retrieval config change.

Run with: python -m eval.compute_retrieval_metrics
"""

import statistics
from pathlib import Path

from app.pipelines.plain.retrieval import PlainRetriever
from eval.span_scoring import ChunkSpanLookup, load_qrels_spans, precision_recall


def main() -> None:
    qrels = load_qrels_spans()
    retriever = PlainRetriever()
    lookup = ChunkSpanLookup()

    precisions, recalls, f1s = [], [], []
    zero_relevant_questions = []

    print(f"{'Precision':>10} {'Recall':>8} {'Retrieved':>10} {'Relevant':>9}  Question")
    for question, relevant_spans in qrels.items():
        contexts = retriever.retrieve(question)
        pr = precision_recall(contexts, relevant_spans, lookup)
        if pr is None:
            zero_relevant_questions.append(question)
            continue

        precision, recall = pr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        print(f"{precision:>10.2f} {recall:>8.2f} {len(contexts):>10} {len(relevant_spans):>9}  {question[:70]}")

    print(f"\n{'=' * 100}")
    print(f"N = {len(precisions)} questions (excluding {len(zero_relevant_questions)} with zero relevant spans)")
    print(f"Mean Precision: {statistics.mean(precisions):.3f}")
    print(f"Mean Recall:    {statistics.mean(recalls):.3f}  (bounded by the candidate pool, not the full corpus -- see QRELS_JUDGE_INSTRUCTIONS.md)")
    print(f"Mean F1:        {statistics.mean(f1s):.3f}")
    if zero_relevant_questions:
        print("\nQuestions with zero relevant spans (excluded above):")
        for q in zero_relevant_questions:
            print(f"  - {q[:90]}")


if __name__ == "__main__":
    main()
