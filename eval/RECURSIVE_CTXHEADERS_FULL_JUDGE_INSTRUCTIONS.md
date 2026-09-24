# Task: Judge 41 RAG pipeline outputs (retrieval + generated answer)

## Context

You're evaluating one configuration of a retrieval-augmented question-answering
system that answers questions about the EU AI Act (Regulation (EU) 2024/1689)
and related regulations: the "Digital Omnibus" amendment (Regulation (EU)
2026/1744, which changed several of the AI Act's compliance dates), GDPR
(Regulation (EU) 2016/679), and various Commission/AI Office guidance
documents (Codes of Practice, scope guidelines, transparency guidelines, plus
an "AI Act Service Desk" set of per-Article summary pages).

The system retrieves small text chunks from a fixed corpus and uses them to
generate a grounded, cited answer to each question. Each retrieved chunk may
have a short "contextual header" prepended to it (one or two sentences
describing what document/provision the chunk is from) before the chunk's own
text — this is meant to help the retrieval and generation steps understand
what a chunk is about even without surrounding context.

**Input:** `eval/recursive_ctxheaders_full_outputs_for_external_judge.jsonl`
— one JSON object per line, one line per question, 41 questions total. Each
line has:

```json
{
  "question": "...",
  "difficulty": "single-hop | cross-reference | multi-hop | temporal-conflict",
  "expected_answer": "<the correct/golden answer>",
  "expected_source_doc": "<file path of the document that should be cited>",
  "expected_source_section": "<article/section reference within that document>",
  "generated_answer": "<what the pipeline actually answered>",
  "citations": ["<document paths the pipeline cited as sources>"],
  "retrieved_chunks": [
    {"rank": 1, "source_doc": "...", "chunk_index": 0, "score": 0.83, "text": "<chunk text, including its prepended contextual header if any>"},
    ...
  ]
}
```

## What to score, per question

Judge using **only** the `retrieved_chunks` (exactly as the pipeline saw
them) and the `generated_answer` — not your own outside knowledge of these
regulations — against the `expected_answer` / `expected_source_doc` /
`expected_source_section`.

1. **`retrieval_hit`** (boolean) — is a chunk from `expected_source_doc`
   present anywhere in `retrieved_chunks`?
2. **`chunk_level_hit`** (boolean) — beyond just the right document, does at
   least one retrieved chunk actually contain the specific fact/provision
   described in `expected_answer` (not just the same document generally)?
   Stricter than `retrieval_hit` — this catches "right document, wrong
   chunk" misses.
3. **`mrr`** (float, 0-1) — reciprocal rank (1 / rank) of the first chunk
   that satisfies `chunk_level_hit`; 0 if none do.
4. **`faithfulness`** (float, 0-1) — is every claim in `generated_answer`
   actually supported by the retrieved chunks (not invented, not from
   outside knowledge)? 1.0 = fully grounded, 0.0 = unsupported/hallucinated.
5. **`answer_correctness`** (float, 0-1) — does `generated_answer` match
   `expected_answer` in substance (not exact wording)? Give partial credit
   for a correct top-level answer that drops a material exception/carve-out
   the golden answer treats as essential.
6. **`citation_accuracy`** (float, 0-1) — do the `citations` the pipeline
   returned actually reflect what supports the answer, or do they just list
   every retrieved document regardless of whether it was actually used?
   1.0 = citations precisely match what's used/needed; lower scores for
   citing unused/irrelevant documents, or for missing a document the answer
   actually relies on.
7. **`outcome_label`** — exactly one of: `Correct`, `Correct-Partial`,
   `Correct-Condensed`, `Partial`, `Abstained-Justified`, `Task-Fail`,
   `Incorrect`, `Hallucinated`.
8. **`comment`** (short free text) — anything notable: a wrong chunk
   retrieved but the answer got lucky anyway; a wrong statutory citation
   (e.g. citing the wrong article/paragraph number even when the
   *substance* of the answer is correct) — flag this specifically and
   quote the exact wrong citation if you find one; self-inconsistency with
   how a similar question elsewhere in this set was answered; boilerplate
   or navigation text diluting a retrieved chunk; a question that needed
   information from two different documents where only one was actually
   retrieved.

## Output format

One JSONL file, one line per question, same order as the input:

```json
{"question": "...", "retrieval_hit": true, "chunk_level_hit": true, "mrr": 1.0, "faithfulness": 0.9, "answer_correctness": 0.8, "citation_accuracy": 0.7, "outcome_label": "Correct-Partial", "comment": "..."}
```

Save as `recursive_ctxheaders_full_judge_scores.jsonl` and return it.

Also write a short (roughly 400-600 word) plain-text or markdown summary of
overall findings: aggregate rates/means for each of the 6 numeric/boolean
metrics, the distribution of `outcome_label` values, and in particular —
**whether any question shows the model citing a specific article/paragraph
number that doesn't match the actual provision it's describing**, even when
the surrounding explanation is substantively correct. That specific failure
pattern (right idea, wrong citation number) is the main thing this
evaluation round is checking for.
