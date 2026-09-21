"""Runs the Groq-free, judge-free retrieval-hit check as a live Langfuse
Experiment against the 'eu-ai-act-golden-set' Dataset. Unlike
langfuse_backfill_experiments.py (which replays already-saved scores from
disk), this makes live retrieval calls — but no LLM calls of any kind (no
generation, no judge), so it's still allowed by the Groq-restriction rule in
DECISIONS.md. Use this for any new chunking/retrieval config change that only
needs the retrieval-hit signal, so the Compare Experiments view stays current
without waiting for a full Groq/Sonnet-judged run.

Run with: python -m eval.langfuse_retrieval_hit_experiment <run_name> <description>
"""

import sys

from langfuse import Langfuse

from app.core import config
from app.pipelines.plain.retrieval import PlainRetriever

DATASET_NAME = "eu-ai-act-golden-set"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python -m eval.langfuse_retrieval_hit_experiment <run_name> <description>")
    run_name, description = sys.argv[1], sys.argv[2]

    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        raise RuntimeError("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set in .env")

    client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
    )
    dataset = client.get_dataset(DATASET_NAME)
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
    )
    print(f"-> {run_name}: {len(result.item_results)} items")
    client.flush()


if __name__ == "__main__":
    main()
