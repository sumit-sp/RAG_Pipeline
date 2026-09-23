"""Optional golden-set / qrels lookup for the experimentation UI.

If the typed question matches one already in eval/golden_set.jsonl or
eval/qrels.jsonl, show the known-good answer and compute instant, real
Precision/Recall for that single question against qrels -- same definition
as eval/compute_retrieval_metrics.py, just for one question at a time
instead of the whole set.
"""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent.parent / "eval"
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.jsonl"
QRELS_PATH = EVAL_DIR / "qrels.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_golden_set() -> dict[str, dict]:
    return {item["question"]: item for item in _load_jsonl(GOLDEN_SET_PATH)}


def load_qrels() -> dict[str, set[tuple[str, int]]]:
    qrels = {}
    for item in _load_jsonl(QRELS_PATH):
        qrels[item["question"]] = {
            (rc["source_doc"], rc["chunk_index"]) for rc in item["relevant_chunks"]
        }
    return qrels


def precision_recall_f1(
    retrieved: set[tuple[str, int]], relevant: set[tuple[str, int]]
) -> tuple[float, float, float] | None:
    if not relevant:
        return None
    hits = retrieved & relevant
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    recall = len(hits) / len(relevant)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1
