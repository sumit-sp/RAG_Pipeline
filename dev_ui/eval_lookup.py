"""Golden-set lookup for the experimentation UI: if the typed question
matches one already in eval/golden_set.jsonl, show the known-good answer
alongside the live one. Precision/Recall scoring itself lives in
eval/span_scoring.py (character-span-based, works for any chunking
strategy) -- re-exported here for the UI's convenience.
"""

import json
from pathlib import Path

from eval.span_scoring import ChunkSpanLookup, load_qrels_spans, precision_recall  # noqa: F401 (re-exported)

GOLDEN_SET_PATH = Path(__file__).parent.parent / "eval" / "golden_set.jsonl"


def load_golden_set() -> dict[str, dict]:
    if not GOLDEN_SET_PATH.exists():
        return {}
    return {
        json.loads(line)["question"]: json.loads(line)
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
