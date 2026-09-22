# Task: Judge all 41 contextual-headers pipeline outputs

**Input:** `eval/contextual_headers_full_outputs_for_external_judge.jsonl` —
one JSON object per line, one line per question, **all 41 golden-set
questions** (not the 15-question hard subset from Step 8). Each line has:

```json
{
  "question": "...",
  "difficulty": "single-hop | cross-reference | multi-hop | temporal-conflict",
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

Generated against the current pipeline: hybrid retrieval (dense + BM25) +
fixed 500/50 chunking + Sonnet-5-generated contextual chunk headers baked into
every chunk before embedding (see `EVALUATION_HISTORY.md` Step 7). This is the
same rubric already used for the 15-question hard subset (Step 8) — judging
the full 41 now gives a complete, directly comparable picture instead of just
the hardest tail.

## What to score, per question

For each record, judge using **only** the `retrieved_chunks` (as the pipeline
actually saw them, headers included) and the `generated_answer` — not outside
knowledge — against the `expected_answer`/`expected_source_doc`/
`expected_source_section`:

1. **`retrieval_hit`** (boolean) — is a chunk from `expected_source_doc` present
   anywhere in `retrieved_chunks`?
2. **`chunk_level_hit`** (boolean) — beyond just the right document, does at
   least one retrieved chunk actually contain the specific fact/provision in
   `expected_answer` (not just the same document generally)? Stricter than
   `retrieval_hit` — catches "right document, wrong chunk" misses.
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
   every retrieved document regardless of whether it was used? 1.0 = citations
   precisely match what's actually used/needed; lower scores for citing
   unused/irrelevant documents or missing a document the answer actually
   relies on. (Every prior judging round found this mechanically echoes the
   retrieved set rather than reflecting real support — worth checking whether
   that's still true across the full set.)
7. **`outcome_label`** (one of: `Correct`, `Correct-Partial`,
   `Correct-Condensed`, `Partial`, `Abstained-Justified`, `Task-Fail`,
   `Incorrect`, `Hallucinated`) — same taxonomy as prior judge rounds, for
   continuity.
8. **`comment`** (short free text) — anything notable: a wrong chunk retrieved
   but the answer got lucky, self-inconsistency with how a similar question
   was answered elsewhere in this set, boilerplate/navigation noise in the
   retrieved text, a GDPR cross-reference question where the GDPR side wasn't
   retrieved at all (see `EVALUATION_HISTORY.md` Step 9 for the known pattern
   there), etc.

## Output format

One JSONL file, one line per question, same order as the input:

```json
{"question": "...", "retrieval_hit": true, "chunk_level_hit": true, "mrr": 1.0, "faithfulness": 0.9, "answer_correctness": 0.8, "citation_accuracy": 0.7, "outcome_label": "Correct-Partial", "comment": "..."}
```

Save as `eval/contextual_headers_full_judge_scores.jsonl` (41 lines expected)
and bring it back. Also write a short (roughly 400-600 word) plain-text or
markdown summary of overall findings, same spirit as prior rounds
(`EVALUATION_HISTORY.md` Steps 6 and 8) — in particular:
- whether the easier ~26 single-hop questions (not covered by the Step 8 hard
  subset) hold up as well as the Groq-free retrieval-hit screen suggested
  (36/41, 87.8%), or if judging surfaces problems the reference-based check
  can't see;
- whether the specific findings from Step 8 (doc-path scoring false
  negatives, inconsistent GDPR cross-reference retrieval, the Article 54(6)
  vs. 53(2) self-inconsistency, low citation accuracy, boilerplate noise)
  reproduce, worsen, or don't reappear across the full set.
