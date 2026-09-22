"""Splits eval/qrels_candidates.jsonl (all 41 questions, ~1370 candidate
chunks total, ~4MB) into smaller batches so each one is a manageable size for
careful per-chunk relevance judging in a single Sonnet-5 session, rather than
one enormous file that encourages skimming instead of real judgment.

Run with: python -m eval.split_qrels_candidates
"""

import json
from pathlib import Path

INPUT_PATH = Path(__file__).parent / "qrels_candidates.jsonl"
OUTPUT_DIR = Path(__file__).parent / "qrels_batches"
QUESTIONS_PER_BATCH = 6


def main() -> None:
    records = [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    OUTPUT_DIR.mkdir(exist_ok=True)
    num_batches = (len(records) + QUESTIONS_PER_BATCH - 1) // QUESTIONS_PER_BATCH

    for batch_num in range(num_batches):
        batch = records[batch_num * QUESTIONS_PER_BATCH : (batch_num + 1) * QUESTIONS_PER_BATCH]
        path = OUTPUT_DIR / f"batch_{batch_num + 1:02d}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for record in batch:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        total_candidates = sum(len(r["candidates"]) for r in batch)
        print(
            f"{path.name}: {len(batch)} questions, {total_candidates} candidate chunks, "
            f"{path.stat().st_size / 1024:.0f} KB"
        )

    print(f"\nWrote {num_batches} batches to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
