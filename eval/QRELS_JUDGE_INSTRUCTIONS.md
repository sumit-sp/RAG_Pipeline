# Task: Build a relevance-judgment set (qrels) for retrieval evaluation

**Why this exists:** every retrieval metric this project has used so far
(Retrieval Hit, Chunk-Level Hit, MRR) checks against a single canonical
"expected" location per question — never against a complete list of every
chunk that's actually relevant. That means none of them are true
Precision@k or Recall@k. This task builds that missing ground truth: for
each question, the full set of chunks (out of a candidate pool, not the
whole 837-chunk corpus — see "Scope" below) that a retriever *should* be
credited for finding.

**Input:** one of `eval/qrels_batches/batch_01.jsonl` through `batch_07.jsonl`
— process **one batch per session** (each is ~500-625KB, 5-6 questions, ~170-215
candidate chunks). Do not try to do all 7 in one sitting; each deserves careful
per-chunk reading, not skimming. One JSON object per line:

```json
{
  "question": "...",
  "difficulty": "single-hop | cross-reference | multi-hop | temporal-conflict",
  "expected_answer": "<golden answer>",
  "expected_source_doc": "regulation/ai_act_2024_1689.html",
  "expected_source_section": "<article/section reference>",
  "candidates": [
    {"source_doc": "...", "chunk_index": 0, "text": "<chunk text, including its prepended contextual header>"},
    ...
  ]
}
```

## Scope — read this before judging anything

The `candidates` list for each question is **not** the full 837-chunk corpus
— it's the union of the top-20 results from 4 independent retrieval methods
(dense-only, sparse-only, hybrid, and the live pipeline with all current
boosts), de-duplicated. This is deliberate: exhaustively judging all 837
chunks against all 41 questions (~34,000 judgments) isn't tractable, and
standard IR practice ("pooling") is to judge the union of several
independent methods' results instead, on the assumption that a genuinely
relevant chunk is very likely to surface in at least one of them.

**Consequence to be aware of, honestly:** any recall computed against this
file later is bounded by this pool — a relevant chunk that none of the 4
methods ever surfaced won't be in `candidates` at all, so it can't be
credited either way. This is a real, accepted limitation of pooling, not
something to work around; just don't be surprised if a future recall number
looks unexpectedly close to 100% — it's recall *within the pool*, not recall
against the full corpus.

## What "relevant" means — read this carefully, it's the part most likely to go wrong

Mark a chunk relevant **only if its content would actually be used or needed
to construct or verify the golden answer** — not just because it's from the
same regulation, mentions the same general topic, or is in the neighborhood
of the right article.

This distinction just caused real, measured harm in this project (see
`EVALUATION_HISTORY.md` Step 13): a retrieval boost that fired on "this
chunk merely *names* another regulation" produced citation-accuracy damage
and even a generation regression, precisely because topical proximity was
being confused with actual relevance. Don't repeat that mistake here.

- **Relevant:** a chunk containing the specific fact, date, obligation,
  exception, or provision the `expected_answer` states — even if phrased
  differently, in a different document than `expected_source_doc`, or only
  partially covering the answer (e.g. one of two facts a multi-hop question
  needs).
- **Not relevant:** a chunk from the right document or the right general
  area of law that doesn't actually contain anything needed for *this
  specific* answer — recitals that mention the same regulation in passing,
  boilerplate cross-references, adjacent articles that don't bear on the
  question asked, or Service Desk navigation/table-of-contents text.
- **When in doubt:** ask "if this were the *only* chunk retrieved, would it
  let the answer be constructed or meaningfully supported?" If no, it's not
  relevant, even if it's thematically close.

## Output format

For each question, list only the chunks you judged **relevant** — chunks not
listed are implicitly judged not relevant, so there's no need to write out a
"not relevant" entry for the majority of the pool that doesn't qualify. One
JSON object per line, same order as the input, one line per question:

```json
{"question": "...", "relevant_chunks": [{"source_doc": "...", "chunk_index": 0, "reason": "states the exact date the question asks about"}, ...]}
```

`reason` should be one short phrase — just enough for a sanity check later,
not a full explanation. If a question has zero relevant chunks in its pool
(a real possibility, and itself a useful finding), use `"relevant_chunks": []`.

Save each batch's output as `eval/qrels_batches/batch_NN_judged.jsonl`
(matching the input batch number, e.g. `batch_01_judged.jsonl` for
`batch_01.jsonl`) and bring all 7 back together once done.
