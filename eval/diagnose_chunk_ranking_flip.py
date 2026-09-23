"""Step 18 systematic diagnostic, part 3: for the specific question where
recursive+headers cited the wrong article, is the wrong chunk
(article_55.html) outranking the correct one (article_53.html) because of
the dense embedding, the sparse (BM25) score, or the RRF fusion of both?

No LLM calls. Runs a dense-only search, a sparse-only search, and the real
hybrid (RRF) search against the same collection, and prints each
candidate chunk's rank/score in all three, so the two chunks of interest
can be compared directly instead of just seeing the final fused ranking.

Run with: python -m eval.diagnose_chunk_ranking_flip
"""

from app.core import config
from app.pipelines.plain.retrieval import PlainRetriever

QUESTION = (
    "The Commission's GPAI scope guidelines describe an open-source exemption from "
    "Article 53(1)(a)-(b). Which article of the AI Act's own text, as also captured "
    "in the Service Desk's Article 53 page, establishes this exemption, and what "
    "condition removes it?"
)

# Corpus where the wrong citation ("Article 54(6)") was observed (Step 18).
CORPUS_BASE_NAME = "ai_act_corpus_recursive_ctxheaders"

# The two chunks whose rank flip we're explaining (found manually in Step 18).
CHUNKS_OF_INTEREST = {
    ("guidance/service_desk_articles/article_55.html", 9): "wrong (outranked correct chunk)",
    ("guidance/service_desk_articles/article_53.html", 0): "correct",
}

CANDIDATE_LIMIT = 30


def main() -> None:
    config.QDRANT_COLLECTION = CORPUS_BASE_NAME
    retriever = PlainRetriever()
    collection = config.collection_name()

    dense_vector = retriever.embedder.embed_query(QUESTION)
    sparse_vector = retriever.sparse_embedder.embed_query(QUESTION)

    dense_points = retriever.client.query_points(
        collection_name=collection, query=dense_vector, using="dense", limit=CANDIDATE_LIMIT
    ).points
    sparse_points = retriever.client.query_points(
        collection_name=collection, query=sparse_vector, using="sparse", limit=CANDIDATE_LIMIT
    ).points
    fused_contexts = retriever._search(QUESTION, limit=CANDIDATE_LIMIT)

    def _rank_map(points):
        # id -> (rank (1-based), score)
        return {str(p.id): (i + 1, p.score) for i, p in enumerate(points)}

    dense_ranks = _rank_map(dense_points)
    sparse_ranks = _rank_map(sparse_points)
    fused_ranks = {c.chunk.id: (i + 1, c.score) for i, c in enumerate(fused_contexts)}

    # id -> (source_doc, chunk_index) via the fused contexts (has full payload)
    id_to_key = {c.chunk.id: (c.chunk.source_doc, c.chunk.chunk_index) for c in fused_contexts}
    # also map from dense/sparse-only results, since some of those points may not appear in fused top-k
    for p in dense_points + sparse_points:
        if str(p.id) not in id_to_key:
            id_to_key[str(p.id)] = (p.payload["source_doc"], p.payload["chunk_index"])

    print(f"Question: {QUESTION[:90]}...")
    print(f"Corpus: {CORPUS_BASE_NAME} -> collection '{collection}'\n")

    print(f"{'source_doc':45s} {'idx':>4s}  {'dense rank':>10s} {'dense score':>11s}  "
          f"{'sparse rank':>11s} {'sparse score':>12s}  {'fused rank':>10s} {'fused score':>11s}  note")
    print("-" * 150)

    all_ids = set(dense_ranks) | set(sparse_ranks) | set(fused_ranks)
    rows = []
    for pid in all_ids:
        key = id_to_key.get(pid, ("?", -1))
        dr, ds = dense_ranks.get(pid, (None, None))
        sr, ss = sparse_ranks.get(pid, (None, None))
        fr, fs = fused_ranks.get(pid, (None, None))
        note = CHUNKS_OF_INTEREST.get(key, "")
        rows.append((key, dr, ds, sr, ss, fr, fs, note))

    # Show the two chunks of interest first, then the rest sorted by fused rank (unranked last)
    rows.sort(key=lambda r: (r[6] is None, r[6] if r[6] is not None else 999, r[0] != list(CHUNKS_OF_INTEREST)[0]))

    for (source_doc, idx), dr, ds, sr, ss, fr, fs, note in rows:
        dr_s = f"{dr}" if dr else "-"
        ds_s = f"{ds:.4f}" if ds is not None else "-"
        sr_s = f"{sr}" if sr else "-"
        ss_s = f"{ss:.4f}" if ss is not None else "-"
        fr_s = f"{fr}" if fr else "-"
        fs_s = f"{fs:.4f}" if fs is not None else "-"
        marker = "  <<<" if note else ""
        print(f"{source_doc:45s} {idx:>4d}  {dr_s:>10s} {ds_s:>11s}  {sr_s:>11s} {ss_s:>12s}  "
              f"{fr_s:>10s} {fs_s:>11s}  {note}{marker}")


if __name__ == "__main__":
    main()
