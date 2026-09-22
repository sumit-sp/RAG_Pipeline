# Task: Judge 15 hard-subset RAG pipeline outputs

**Input:** `eval/hard_subset_outputs_for_external_judge.jsonl` — one JSON object
per line, one line per question (15 total: 8 cross-reference, 3 multi-hop, 2
temporal-conflict, 2 single-hop questions that fail the automated retrieval
check). Each line has:

```json
{
  "question": "...",
  "difficulty": "cross-reference",
  "expected_answer": "<golden answer>",
  "expected_source_doc": "regulation/ai_act_2024_1689.html",
  "expected_source_section": "<article/section reference>",
  "generated_answer": "<what the pipeline actually answered>",
  "citations": ["<source_doc paths the pipeline cited>"],
  "retrieved_chunks": [
    {"rank": 1, "source_doc": "...", "chunk_index": 0, "score": 0.83, "text": "<chunk text, including its prepended contextual header>"},
    ...
  ]
}
```

This is the **hard subset** — every question likely to actually fail
(cross-reference, multi-hop, temporal-conflict, plus 2 single-hop questions
already known to miss retrieval), not the full 41-question golden set. It was
generated against the current pipeline: hybrid retrieval (dense + BM25) +
fixed 500/50 chunking + Sonnet-5-generated contextual chunk headers baked into
every chunk before embedding.

## What to score, per question

For each record, judge using **only** the `retrieved_chunks` (as the pipeline
actually saw them, headers included) and the `generated_answer` — not outside
knowledge — against the `expected_answer`/`expected_source_doc`/
`expected_source_section`:

1. **`retrieval_hit`** (boolean) — is a chunk from `expected_source_doc` present
   anywhere in `retrieved_chunks`?
2. **`chunk_level_hit`** (boolean) — beyond just the right document, does at
   least one retrieved chunk actually contain the specific fact/provision in
   `expected_answer` (not just the same document generally)? This is stricter
   than `retrieval_hit` and catches "right document, wrong chunk" misses.
3. **`mrr`** (float, 0-1) — reciprocal rank (1/rank) of the first chunk that
   satisfies `chunk_level_hit`; 0 if none do.
4. **`faithfulness`** (float, 0-1) — is every claim in `generated_answer`
   actually supported by the retrieved chunks (not invented, not from outside
   knowledge)? 1.0 = fully grounded, 0.0 = unsupported/hallucinated.
5. **`answer_correctness`** (float, 0-1) — does `generated_answer` match
   `expected_answer` in substance (not wording)? Partial credit for a correct
   top-level answer that drops a material exception/carve-out the golden
   answer treats as essential.
6. **`citation_accuracy`** (float, 0-1) — do the `citations` the pipeline
   returned actually reflect what supports the answer, or do they just list
   every retrieved document regardless of whether it was used? (This is a
   known open finding from the last judge round — citations previously just
   echoed the retrieved set.) 1.0 = citations precisely match what's actually
   used/needed; lower scores for citing unused/irrelevant documents or missing
   a document the answer actually relies on.
7. **`outcome_label`** (one of: `Correct`, `Correct-Partial`,
   `Correct-Condensed`, `Partial`, `Abstained-Justified`, `Task-Fail`,
   `Incorrect`, `Hallucinated`) — same taxonomy as the previous judge round,
   for continuity.
8. **`comment`** (short free text) — anything notable: a wrong chunk retrieved
   but the answer got lucky, self-inconsistency with how a similar question
   was answered elsewhere in this set, boilerplate/navigation noise in the
   retrieved text, etc.

## Output format

One JSONL file, one line per question, same order as the input:

```json
{"question": "...", "retrieval_hit": true, "chunk_level_hit": true, "mrr": 1.0, "faithfulness": 0.9, "answer_correctness": 0.8, "citation_accuracy": 0.7, "outcome_label": "Correct-Partial", "comment": "..."}
```

Save as `eval/hard_subset_judge_results.jsonl` (15 lines expected) and bring it
back. Also write a short (roughly 300-500 word) plain-text or markdown summary
of overall findings — same spirit as the narrative write-up from the last
judge round (`EVALUATION_HISTORY.md` Step 6) — highlighting any new patterns
specific to this harder question set, especially anything relevant to the
still-open findings: chunk-level recall vs. document-level recall, citation
accuracy, and cross-reference/multi-hop weakness.
