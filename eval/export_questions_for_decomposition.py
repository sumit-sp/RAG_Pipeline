"""Exports the golden set's multi-hop questions for external decomposition
into sub-questions (Step 19 -- query decomposition experiment).

Per the standing rule (Groq restricted to generation only; any other
LLM-content task is exported for an external Sonnet-5 session, never done
live in this repo -- see DECISIONS.md's rejection of live query
reformulation/HyDE for the same reason), decomposition happens once,
offline, here -- not as a live call at retrieval time.

Deliberately excludes expected_answer/expected_source_doc from the export:
decomposition should work from the question's natural language alone, the
way a real system would face it, not lean on ground truth that a live
system wouldn't have.

Run with: python -m eval.export_questions_for_decomposition
"""

import json
from pathlib import Path

EVAL_DIR = Path(__file__).parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.jsonl"
OUTPUT_PATH = EVAL_DIR / "questions_for_decomposition.jsonl"


def main() -> None:
    items = [
        json.loads(line)
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    multi_hop = [item for item in items if item.get("difficulty") == "multi-hop"]

    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for item in multi_hop:
            out.write(json.dumps({"question": item["question"]}) + "\n")

    print(f"Exported {len(multi_hop)} multi-hop questions -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
