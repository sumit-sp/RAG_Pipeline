"""Runs the deterministic Precision@k/Recall@k check (eval/qrels.jsonl) as a
live Langfuse Experiment against the 'eu-ai-act-golden-set' Dataset. Like
langfuse_retrieval_hit_experiment.py, this makes live retrieval calls but no
LLM calls of any kind, so it's allowed by the Groq-restriction rule.

Run with: python -m eval.langfuse_precision_recall_experiment <run_name> <description> [metadata_overrides_json]
"""

import json
import sys
from pathlib import Path

from langfuse import Langfuse

from app.core import config
from app.pipelines.plain.retrieval import PlainRetriever
from eval.pipeline_metadata import build_pipeline_metadata

DATASET_NAME = "eu-ai-act-golden-set"
QRELS_PATH = Path(__file__).parent / "qrels.jsonl"


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
    if len(sys.argv) not in (3, 4):
        raise SystemExit(
            "Usage: python -m eval.langfuse_precision_recall_experiment <run_name> <description> [metadata_overrides_json]"
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
    run_metadata = build_pipeline_metadata(overrides)  # computed before the retriever opens its own client
    qrels = _load_qrels()
    retriever = PlainRetriever()

    def task(*, item, **kwargs):
        contexts = retriever.retrieve(item.input)
        return sorted({(c.chunk.source_doc, c.chunk.chunk_index) for c in contexts})

    def evaluator(*, input, output, expected_output, metadata, **kwargs):
        relevant = qrels.get(input, set())
        if not relevant:
            return []
        retrieved = {tuple(pair) for pair in (output or [])}
        hits = retrieved & relevant
        precision = len(hits) / len(retrieved) if retrieved else 0.0
        recall = len(hits) / len(relevant)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return [
            {"name": "Precision", "value": precision},
            {"name": "Recall", "value": recall},
            {"name": "F1", "value": f1},
        ]

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
