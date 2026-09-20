# Eval results

All numbers below are against the same 41 questions in `eval/golden_set.jsonl`, via DeepEval,
judged by Groq `gpt-oss-120b` (see `DECISIONS.md`). Metric threshold: 0.5. Raw per-question
scores for each run are in `eval/eval_run_raw_results_<mode>.jsonl`.

**Read this before trusting a decimal place:** identical-config re-runs of this suite show
real run-to-run variance (~0.05-0.07 on individual metric means), because Groq's served LLM
inference isn't bit-exact at `temperature=0`, and that affects both the generator and the
judge model. See "run-to-run variance" in `DECISIONS.md`. Treat the overall pass/fail count as
the primary signal; treat single-metric deltas smaller than ~0.07 as within the noise floor.

## Phase 2 baseline (dense-only retrieval, original)

First baseline run, before a bug was found and fixed (see Phase 3 below): 26/41 passed, and
2/41 questions crashed out of scoring entirely (empty Groq response). Superseded by the
bug-fixed re-run below — kept here only as the historical Phase 2 record.

| Metric | Mean (n=39, excl. 2 crashes) |
|---|---|
| Faithfulness | 0.952 |
| Answer Relevancy | 0.977 |
| Contextual Precision | 0.759 |
| Contextual Recall | 0.752 |

## Phase 3

### Fix: empty Groq responses (finish_reason="length" on gpt-oss reasoning models)

Not a retrieval/generation *quality* change, but a robustness fix required before any
before/after comparison could be trusted — see `DECISIONS.md`. Re-running dense-only after the
fix, all 41 questions completed (0 crashes, vs. 2/41 before):

| Metric | Threshold | Mean | Pass rate |
|---|---|---|---|
| Faithfulness | 0.5 | 0.887 | 37/41 |
| Answer Relevancy | 0.5 | 0.989 | 41/41 |
| Contextual Precision | 0.5 | 0.691 | 31/41 |
| Contextual Recall | 0.5 | 0.772 | 33/41 |
| **Overall (all 4 pass)** | | | **27/41 (66%)** |

This is the real "current best" baseline Phase 3 changes are measured against — not the
Phase 2 numbers above, which included 2 crashed questions and predate the fix.

### Tried: hybrid search (dense + BM25 sparse, RRF fusion) — KEPT

| Metric | Threshold | Mean | Pass rate | Δ vs. dense (post-fix) |
|---|---|---|---|---|
| Faithfulness | 0.5 | 0.931 | 39/41 | +0.044 |
| Answer Relevancy | 0.5 | 0.917 | 38/41 | −0.072 |
| Contextual Precision | 0.5 | 0.754 | 35/41 | +0.063 |
| Contextual Recall | 0.5 | 0.776 | 34/41 | +0.004 |
| **Overall (all 4 pass)** | | | **31/41 (76%)** | **+4 questions** |

By difficulty (contextual precision / recall, dense → hybrid):

| Difficulty | n | Precision | Recall |
|---|---|---|---|
| single-hop | 28 | 0.712 → 0.742 | 0.815 → 0.845 |
| cross-reference | 8 | 0.770 → 0.826 | 0.792 → 0.667 |
| multi-hop | 3 | 0.667 → 0.844 | 0.500 → 0.611 |
| temporal-conflict | 2 | 0.125 → 0.500 | 0.500 → 0.500 |

**Kept.** The overall pass-rate swing (27→31, +4 questions) is well above the observed noise
floor, and the mechanism makes sense: dense embeddings match on semantic topic similarity,
which can miss exact-term hits (article numbers, defined terms) that BM25 catches directly —
precisely what multi-hop and cross-reference questions need, since they require chunks pulled
in by different signals from two different documents. Answer relevancy dropped somewhat
(0.989→0.917) but stayed well clear of threshold; not treated as a regression worth reverting
for. `RETRIEVAL_MODE=hybrid` is now the default (see `DECISIONS.md`).

Cross-reference recall (0.792→0.667) went the "wrong" way despite precision improving a lot —
plausible explanation: RRF fusion's ranking can promote a highly BM25-relevant chunk ahead of
an equally-needed second chunk that only the dense signal was finding, effectively trading
recall for precision within a fixed top-k. Worth revisiting if reranking (tried next) doesn't
also fix it.

### Tried: cross-encoder reranking on top of hybrid search — REVERTED

| Metric | Hybrid (current best) | +Reranking | Δ |
|---|---|---|---|
| Faithfulness | 0.931 | 0.935 | +0.004 |
| Answer Relevancy | 0.917 | 0.984 | +0.067 |
| Contextual Precision | 0.754 | 0.718 | −0.036 |
| Contextual Recall | 0.776 | 0.809 | +0.033 |
| **Overall (all 4 pass)** | **31/41 (76%)** | **26/41 (63%)** | **−5 questions** |

By difficulty (precision / recall, hybrid → +reranking):

| Difficulty | n | Precision | Recall |
|---|---|---|---|
| single-hop | 28 | 0.742 → 0.685 | 0.845 → 0.851 |
| cross-reference | 8 | 0.826 → 0.878 | 0.667 → 0.792 |
| multi-hop | 3 | 0.844 → 0.602 | 0.611 → 0.500 |
| temporal-conflict | 2 | 0.500 → 0.725 | 0.500 → 0.750 |

**Reverted.** Even though 2 of 4 metrics improved and two difficulty buckets got better, the
overall pass count dropped by 5 questions — a swing well past the noise floor — because the
metrics that got worse (precision, and multi-hop specifically) knocked out questions that
weren't failing before. `cross-encoder/ms-marco-MiniLM-L-6-v2` is trained on general web
passage ranking, not legal/regulatory text; it appears to work against the hybrid retrieval's
exact-term (BM25) signal rather than refine it, especially where a question needs two specific
documents together (multi-hop). `USE_RERANKING` defaults to `false`; code kept for a possible
future attempt with a different reranker. See `DECISIONS.md`.

### Not yet tried

- Contextual chunk headers
- Chunk size / overlap tuning
