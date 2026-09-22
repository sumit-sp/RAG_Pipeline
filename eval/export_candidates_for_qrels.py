"""One-off export (not part of the permanent app): builds a per-question
candidate pool for Claude Sonnet 5 to label as a true relevance-judgment set
("qrels", the standard IR term) — see EVALUATION_HISTORY.md for why this is
needed: none of the existing metrics (Retrieval Hit, Chunk-Level Hit, MRR)
are real Precision@k/Recall@k, because none of them are checked against an
exhaustive list of every relevant chunk for a question, only a single
canonical `expected_source_doc` pointer.

Exhaustively judging relevance against all 837 corpus chunks for all 41
questions isn't tractable in one sitting (that's ~34,000 judgments) and isn't
how real IR relevance judgments get built anyway — the standard practice
("pooling") is to union the top-N results from several independent retrieval
methods and judge only that pool, on the assumption that a truly relevant
chunk is highly likely to surface in at least one method's top results. This
script builds that pool from 4 independent sources per question:
dense-only, sparse-only, hybrid (RRF, no boost), and the actual live
pipeline (hybrid + contextual headers + cross-reference boost) — so the pool
isn't biased toward whichever config we're trying to evaluate.

Caveat this creates, stated for the record: recall computed against this
qrels file is bounded by the pool, not the full corpus — a truly relevant
chunk that none of these 4 methods ever surfaces would be invisible to this
process. This is the standard, accepted trade-off in real-world IR
evaluation, not a shortcut unique to this project.

No LLM calls here — pure retrieval mechanics, same as
eval/diagnose_gdpr_retrieval.py. Judging happens externally (Sonnet 5).

Run with: python -m eval.export_candidates_for_qrels
"""

import json
from pathlib import Path

from qdrant_client.models import FusionQuery, Prefetch

from app.core import config
from app.pipelines.plain.retrieval import PlainRetriever

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
OUTPUT_PATH = Path(__file__).parent / "qrels_candidates.jsonl"
POOL_DEPTH = 20  # top-N pulled from each of the 4 independent sources


def _point_to_dict(p) -> dict:
    return {
        "source_doc": p.payload["source_doc"],
        "chunk_index": p.payload["chunk_index"],
        "text": p.payload["text"],
    }


def main() -> None:
    golden_set = [
        json.loads(line)
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    retriever = PlainRetriever()  # the actual live pipeline: hybrid + headers + boost
    # Reuse the retriever's own client/embedders for the raw dense/sparse/hybrid
    # queries below -- the local (embedded, on-disk) Qdrant mode takes an
    # exclusive file lock per client instance, so a second client in the same
    # process would collide with this one.
    embedder = retriever.embedder
    sparse = retriever.sparse_embedder
    client = retriever.client
    collection = config.collection_name()

    pool_sizes = []
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for i, item in enumerate(golden_set, start=1):
            question = item["question"]
            dv = embedder.embed_query(question)
            sv = sparse.embed_query(question)

            dense_only = client.query_points(
                collection_name=collection, query=dv, using="dense", limit=POOL_DEPTH
            ).points
            sparse_only = client.query_points(
                collection_name=collection, query=sv, using="sparse", limit=POOL_DEPTH
            ).points
            hybrid_no_boost = client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(query=dv, using="dense", limit=config.HYBRID_PREFETCH_LIMIT),
                    Prefetch(query=sv, using="sparse", limit=config.HYBRID_PREFETCH_LIMIT),
                ],
                query=FusionQuery(fusion="rrf"),
                limit=POOL_DEPTH,
            ).points
            live_pipeline = retriever.retrieve(question, top_k=POOL_DEPTH)

            pool: dict[tuple[str, int], dict] = {}
            for p in dense_only:
                pool[(p.payload["source_doc"], p.payload["chunk_index"])] = _point_to_dict(p)
            for p in sparse_only:
                pool[(p.payload["source_doc"], p.payload["chunk_index"])] = _point_to_dict(p)
            for p in hybrid_no_boost:
                pool[(p.payload["source_doc"], p.payload["chunk_index"])] = _point_to_dict(p)
            for c in live_pipeline:
                pool[(c.chunk.source_doc, c.chunk.chunk_index)] = {
                    "source_doc": c.chunk.source_doc,
                    "chunk_index": c.chunk.chunk_index,
                    "text": c.chunk.text,
                }

            candidates = sorted(pool.values(), key=lambda c: (c["source_doc"], c["chunk_index"]))
            pool_sizes.append(len(candidates))

            record = {
                "question": question,
                "difficulty": item["difficulty"],
                "expected_answer": item["expected_answer"],
                "expected_source_doc": item["expected_source_doc"],
                "expected_source_section": item["expected_source_section"],
                "candidates": candidates,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(golden_set)}] pool size {len(candidates)} | {question[:70]}")

    print(
        f"\nWrote {len(golden_set)} records to {OUTPUT_PATH}\n"
        f"Pool sizes: min={min(pool_sizes)} max={max(pool_sizes)} "
        f"mean={sum(pool_sizes) / len(pool_sizes):.1f} total_candidates={sum(pool_sizes)}"
    )


if __name__ == "__main__":
    main()
