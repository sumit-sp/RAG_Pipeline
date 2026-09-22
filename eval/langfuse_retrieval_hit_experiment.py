"""Runs the Groq-free, judge-free retrieval-hit check as a live Langfuse
Experiment against the 'eu-ai-act-golden-set' Dataset. Unlike
langfuse_backfill_experiments.py (which replays already-saved scores from
disk), this makes live retrieval calls — but no LLM calls of any kind (no
generation, no judge), so it's still allowed by the Groq-restriction rule in
DECISIONS.md. Use this for any new chunking/retrieval config change that only
needs the retrieval-hit signal, so the Compare Experiments view stays current
without waiting for a full Groq/Sonnet-judged run.

Every run attaches a full pipeline-config snapshot (chunking, retrieval mode,
embedding model/dimension, etc. — see eval/pipeline_metadata.py) as Langfuse
run metadata, so the Compare Experiments view is self-describing.

Run with: python -m eval.langfuse_retrieval_hit_experiment <run_name> <description> [metadata_overrides_json]

metadata_overrides_json is optional — pass it when the live config flag
doesn't reflect what's actually baked into the collection (e.g.
USE_CONTEXTUAL_HEADERS only affects ingestion, not retrieval), e.g.:
  python -m eval.langfuse_retrieval_hit_experiment my-run "..." '{"use_contextual_headers": true}'
"""

import json
import sys

from langfuse import Langfuse

from app.core import config
from app.pipelines.plain.retrieval import PlainRetriever
from eval.pipeline_metadata import build_pipeline_metadata

DATASET_NAME = "eu-ai-act-golden-set"


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "Usage: python -m eval.langfuse_retrieval_hit_experiment <run_name> <description> [metadata_overrides_json]"
        )
    run_name, description = sys.argv[1], sys.argv[2]
    overrides = json.loads(sys.argv[3]) if len(sys.argv) == 4 else None

    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        raise RuntimeError("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set in .env")

    client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
    )
    dataset = client.get_dataset(DATASET_NAME)
    # Computed before the retriever opens its own Qdrant client -- the local
    # (embedded, on-disk) Qdrant mode takes an exclusive file lock per client
    # instance, so two open at once in this process would collide.
    run_metadata = build_pipeline_metadata(overrides)
    retriever = PlainRetriever()

    def task(*, item, **kwargs):
        contexts = retriever.retrieve(item.input)
        return sorted({c.chunk.source_doc for c in contexts})

    def evaluator(*, input, output, expected_output, metadata, **kwargs):
        hit = metadata["expected_source_doc"] in (output or [])
        return [{"name": "Retrieval Hit", "value": 1.0 if hit else 0.0}]

    result = dataset.run_experiment(
        name=run_name,
        run_name=run_name,
        description=description,
        task=task,
        evaluators=[evaluator],
        metadata=run_metadata,
    )
    print(f"-> {run_name}: {len(result.item_results)} items")
    client.flush()


if __name__ == "__main__":
    main()
