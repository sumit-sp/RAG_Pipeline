"""One-off diagnostic (not part of the permanent app): investigates why GDPR
cross-reference retrieval is inconsistent (EVALUATION_HISTORY.md Step 8,
finding 2) by inspecting dense-only, sparse-only, and hybrid-RRF rank
positions for the GDPR-relevant portion of every retrieved corpus (not just
the final top-5), for the 6 hard-subset questions that involve GDPR.

No LLM calls at all -- pure retrieval-mechanics inspection (embeddings +
Qdrant queries), so it's outside the Groq-restriction rule entirely.

Run with: python -m eval.diagnose_gdpr_retrieval
"""

import json
from pathlib import Path

from qdrant_client.models import FusionQuery, Prefetch

from app.core import config
from app.pipelines.plain.embedding import get_embedder
from app.pipelines.plain.sparse_embedding import SparseEmbedder
from app.pipelines.plain.vector_store import get_qdrant_client

HARD_SUBSET_PATH = Path(__file__).parent / "hard_subset_outputs_for_external_judge.jsonl"

# Indices (0-based, in hard_subset_outputs_for_external_judge.jsonl) of every
# question that involves GDPR, per the Step 8 judge's own commentary.
_GDPR_QUESTION_INDICES = [5, 6, 7, 8, 9, 14]


def _gdpr_rank(points: list, is_gdpr) -> list[int]:
    return [i + 1 for i, p in enumerate(points) if is_gdpr(p.payload["source_doc"])]


def main() -> None:
    recs = [
        json.loads(line)
        for line in HARD_SUBSET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    embedder = get_embedder()
    sparse = SparseEmbedder()
    client = get_qdrant_client()
    collection = config.collection_name()

    def is_real_gdpr(doc: str) -> bool:
        return doc == "adjacent/gdpr_2016_679.html"

    for idx in _GDPR_QUESTION_INDICES:
        q = recs[idx]["question"]
        print("=" * 100)
        print(f"[{idx}] {q}")
        print(f"  expected_source_doc: {recs[idx]['expected_source_doc']}")

        dv = embedder.embed_query(q)
        sv = sparse.embed_query(q)

        dense_only = client.query_points(
            collection_name=collection, query=dv, using="dense", limit=20
        ).points
        sparse_only = client.query_points(
            collection_name=collection, query=sv, using="sparse", limit=20
        ).points
        hybrid = client.query_points(
            collection_name=collection,
            prefetch=[
                Prefetch(query=dv, using="dense", limit=20),
                Prefetch(query=sv, using="sparse", limit=20),
            ],
            query=FusionQuery(fusion="rrf"),
            limit=20,
        ).points

        for label, points in [("dense-only ", dense_only), ("sparse-only", sparse_only), ("hybrid RRF ", hybrid)]:
            ranks = _gdpr_rank(points, is_real_gdpr)
            print(f"  {label}: real-GDPR chunk ranks in top-20 = {ranks if ranks else 'NONE'}")


if __name__ == "__main__":
    main()
