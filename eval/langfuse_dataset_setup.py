"""Creates the Langfuse Dataset from eval/golden_set.jsonl, once. Safe to re-run
— skips creation if the dataset already has items, so it won't duplicate them.

Run with: python -m eval.langfuse_dataset_setup
"""

import json
from pathlib import Path

from langfuse import Langfuse

from app.core import config

DATASET_NAME = "eu-ai-act-golden-set"
GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"


def main() -> None:
    if not (config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        raise RuntimeError("LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set in .env")

    client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
    )

    try:
        existing = client.get_dataset(DATASET_NAME)
        if len(existing.items) > 0:
            print(f"Dataset '{DATASET_NAME}' already has {len(existing.items)} items — skipping.")
            return
    except Exception:
        client.create_dataset(
            name=DATASET_NAME,
            description=(
                "41-question golden set for the EU AI Act compliance assistant "
                "(eval/golden_set.jsonl) — 6 hand-verified temporal-conflict questions "
                "plus 37 agent-drafted, human-reviewed questions across the corpus."
            ),
        )

    items = [
        json.loads(line)
        for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for item in items:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            input=item["question"],
            expected_output=item["expected_answer"],
            metadata={
                "difficulty": item["difficulty"],
                "expected_source_doc": item["expected_source_doc"],
                "expected_source_section": item["expected_source_section"],
            },
        )

    print(f"Created dataset '{DATASET_NAME}' with {len(items)} items")


if __name__ == "__main__":
    main()
