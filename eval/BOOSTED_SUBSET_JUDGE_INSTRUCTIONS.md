# Task: Judge the 15 questions affected by the cross-reference boost (Step 12)

**Input:** `eval/boosted_subset_outputs_for_external_judge.jsonl` — one JSON
object per line, one line per question, for exactly the 15 golden-set
questions where the new cross-reference boost actually fired (out of 41
total — the other 26 are byte-identical to the already-judged Step 10 run,
so they aren't re-exported here). Each line has:

```json
{
  "question": "...",
  "difficulty": "single-hop | cross-reference | multi-hop | temporal-conflict",
  "expected_answer": "<golden answer>",
  "expected_source_doc": "regulation/ai_act_2024_1689.html",
  "expected_source_section": "<article/section reference>",
  "boosted_in_documents": ["adjacent/gdpr_2016_679.html"],
  "generated_answer": "<what the pipeline actually answered>",
  "citations": ["<source_doc paths the pipeline cited>"],
  "retrieved_chunks": [
    {"rank": 1, "source_doc": "...", "chunk_index": 0, "score": 0.83, "text": "<chunk text, including its prepended contextual header>"},
    ...
  ]
}
```

`boosted_in_documents` tells you which document(s) got pulled in by the
boost for that question — usually appearing after rank 5 in
`retrieved_chunks`, since the boost appends extra chunks beyond the normal
top-5. This is exactly what's new and needs checking: **did adding these
extra chunks help, hurt, or do nothing to the generated answer?**

## Why this subset exists

The pipeline itself (retrieval logic, contextual headers, generation model)
didn't change for the other 26 golden-set questions since the last full
judging round — only these 15 got extra retrieved context from the new
boost. Judging just this subset, rather than re-running all 41, avoids
paying judge cost on questions we already know the answer for.

## What to score, per question

Same rubric as every prior round, for direct comparability — judge using
**only** the `retrieved_chunks` (as the pipeline actually saw them, headers
included) and the `generated_answer` — not outside knowledge — against the
`expected_answer`/`expected_source_doc`/`expected_source_section`:

1. **`retrieval_hit`** (boolean) — is a chunk from `expected_source_doc`
   present anywhere in `retrieved_chunks`?
2. **`chunk_level_hit`** (boolean) — does at least one retrieved chunk
   actually contain the specific fact/provision in `expected_answer` (not
   just the same document generally)?
3. **`mrr`** (float, 0-1) — reciprocal rank (1/rank) of the first chunk that
   satisfies `chunk_level_hit`; 0 if none do.
4. **`faithfulness`** (float, 0-1) — is every claim in `generated_answer`
   actually supported by the retrieved chunks?
5. **`answer_correctness`** (float, 0-1) — does `generated_answer` match
   `expected_answer` in substance?
6. **`citation_accuracy`** (float, 0-1) — do the `citations` the pipeline
   returned actually reflect what supports the answer, or do they just list
   every retrieved document regardless of whether it was used?
7. **`outcome_label`** (one of: `Correct`, `Correct-Partial`,
   `Correct-Condensed`, `Partial`, `Abstained-Justified`, `Task-Fail`,
   `Incorrect`, `Hallucinated`).
8. **`comment`** (short free text) — and **specifically address this
   question for every record**: did the boosted-in chunk(s) actually get
   used by the answer, get ignored, or add irrelevant noise/length without
   helping? This is the one new thing this round is trying to measure that
   prior rounds didn't need to.

## Output format

One JSONL file, one line per question, same order as the input:

```json
{"question": "...", "retrieval_hit": true, "chunk_level_hit": true, "mrr": 1.0, "faithfulness": 0.9, "answer_correctness": 0.8, "citation_accuracy": 0.7, "outcome_label": "Correct-Partial", "comment": "..."}
```

Save as `eval/boosted_subset_judge_scores.jsonl` (15 lines expected) and
bring it back. Also write a short (roughly 300-500 word) summary focused
specifically on: across these 15 questions, did the boosted-in context help
more than it hurt? Any case where the extra chunks caused the answer to get
worse (more noise, wrong emphasis, longer but less focused) would be an
important, actionable finding — as important as a case where it helped.
