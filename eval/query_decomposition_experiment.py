"""Step 19: query decomposition experiment, baseline vs. decomposed retrieval,
on the 3 multi-hop golden-set questions -- including the one with the known
Article 54(6)/53(2) citation regression under recursive+headers (Step 18).

Sub-questions come from eval/decomposed_questions.jsonl (externally generated
by Sonnet-5, see DECOMPOSITION_INSTRUCTIONS.md -- no live LLM call here for
decomposition itself, per the standing rule: DECISIONS.md already rejected
live query reformulation/HyDE for needing a Groq call at retrieval time).

For each question, per corpus:
  - baseline: retrieve once for the original question (top_k), generate.
  - decomposed: retrieve once per sub-question (each at top_k), merge +
    dedupe the results, generate against the original question + merged
    context. Uses the SAME per-query top_k as baseline (not a smaller
    per-sub-question budget), so decomposed runs end up using MORE total
    context than baseline -- reported explicitly, not hidden, since that's
    an honest part of the comparison (does more *targeted* context help, or
    just add noise -- the same question Step 13 already had to ask about the
    cross-reference boost).

Precision/Recall come from eval/span_scoring.py (character-span-based, via
eval/qrels_spans.jsonl) -- works for every corpus here, including the
recursive ones, unlike the retired chunk_index-based eval/qrels.jsonl.

One real Groq generation call per (question x corpus x mode) -- allowed,
generation is what the Groq-restriction rule permits.

Run with: python -m eval.query_decomposition_experiment
"""

import json
from pathlib import Path

from app.core import config
from app.core.models import RetrievedContext
from app.pipelines.plain.generation import PlainGenerator
from app.pipelines.plain.retrieval import PlainRetriever
from eval.span_scoring import ChunkSpanLookup, load_qrels_spans, precision_recall

EVAL_DIR = Path(__file__).parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.jsonl"
DECOMPOSED_PATH = EVAL_DIR / "decomposed_questions.jsonl"

# base collection name -> label. RETRIEVAL_MODE stays "hybrid" (default) for all.
CORPORA = {
    "ai_act_corpus": "Production (fixed + real headers)",
    "ai_act_corpus_recursive_ctxheaders": "Recursive + projected headers (Step 18's regression corpus)",
}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _merge_dedupe(context_lists: list[list[RetrievedContext]]) -> list[RetrievedContext]:
    seen_ids = set()
    merged = []
    for contexts in context_lists:
        for c in contexts:
            if c.chunk.id not in seen_ids:
                merged.append(c)
                seen_ids.add(c.chunk.id)
    return merged


def main() -> None:
    if not DECOMPOSED_PATH.exists():
        raise SystemExit(
            f"{DECOMPOSED_PATH} not found -- run eval/export_questions_for_decomposition.py, "
            f"get it decomposed externally per DECOMPOSITION_INSTRUCTIONS.md, and save the "
            f"result there first."
        )

    golden_set = {item["question"]: item for item in _load_jsonl(GOLDEN_SET_PATH)}
    decomposed = {item["question"]: item["sub_questions"] for item in _load_jsonl(DECOMPOSED_PATH)}
    qrels = load_qrels_spans()
    generator = PlainGenerator()
    lookup = ChunkSpanLookup()

    for base_name, corpus_label in CORPORA.items():
        config.QDRANT_COLLECTION = base_name
        config.CHUNKING_STRATEGY = "recursive" if "recursive" in base_name else "fixed"
        retriever = PlainRetriever()
        collection = config.collection_name()
        print(f"\n{'=' * 110}\nCorpus: {corpus_label}  ({collection})\n{'=' * 110}")

        for question, sub_questions in decomposed.items():
            golden = golden_set[question]
            relevant_spans = qrels.get(question, [])

            print(f"\n--- {question[:90]}")
            print(f"    expected_source_doc: {golden['expected_source_doc']}")
            print(f"    sub-questions: {sub_questions}")

            for label, run_context_lists in [
                ("baseline", [retriever.retrieve(question)]),
                ("decomposed", [retriever.retrieve(sq) for sq in sub_questions]),
            ]:
                contexts = _merge_dedupe(run_context_lists)
                retrieved_docs = {c.chunk.source_doc for c in contexts}
                hit = golden["expected_source_doc"] in retrieved_docs
                pr = precision_recall(contexts, relevant_spans, lookup)
                answer = generator.generate(question, contexts)

                pr_str = f"P={pr[0]:.2f} R={pr[1]:.2f}" if pr else "n/a"
                print(f"    [{label:10s}] {len(contexts)} chunks, hit={hit}, {pr_str}")
                print(f"      sources: {', '.join(answer.citations)}")
                print(f"      answer: {answer.text[:280]}")


if __name__ == "__main__":
    main()
