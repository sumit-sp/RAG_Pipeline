# Eval History Summary — 22 Sep 2026

A point-in-time synthesis of everything learned from evaluation so far (Steps 1-10
in `EVALUATION_HISTORY.md`): what's actually working, and what the major gaps are
with respect to the RAG pipeline's overall accuracy. Written before deciding what
to fix next (the GDPR cross-reference retrieval gap, diagnosed in Step 9, was the
next candidate under discussion at the time this was written).

## What's actually working

- **Hybrid search (dense + BM25) is a real, solid win.** +4 questions over
  dense-only, specifically by fixing exact-term matching (article numbers,
  defined terms) that pure semantic embedding missed. Kept as default with high
  confidence — cross-confirmed across every subsequent measurement.
- **Contextual chunk headers are the best single improvement so far.** 87.8%
  retrieval-hit, 82.9% fully/mostly correct on the full 41-question set — and
  this was independently confirmed twice: the Groq-free screen and Claude
  Sonnet 5's full judge landed on the *exact same* 36/41 doc-recall number
  through completely different measurement methods.
- **Single-hop questions are essentially solved.** 93% retrieval hit, 0.950
  answer correctness, 0.971 faithfulness. This is 28 of the 41 questions — the
  majority of the corpus.
- **Faithfulness is generally high** (0.90-0.96 across most runs) — the model
  isn't wildly inventing content when it does answer.
- **Reranking was tried and correctly rejected.** A negative result, but a
  valuable one — it shows the eval discipline actually catches things that look
  good on paper (2 of 4 average metrics improved) but hurt overall pass rate.

## The core, unsolved gap: compound/cross-document questions

This is the single biggest driver of remaining inaccuracy, and no retrieval
improvement so far has touched it:

| Category | Retrieval Hit | Answer Correctness |
|---|---|---|
| single-hop (28 q) | 93% | 0.950 |
| cross-reference (8 q) | 75% | 0.669 |
| multi-hop (3 q) | 67% | 0.633 |
| temporal-conflict (2 q) | 100% | 0.050* |

*driven entirely by generation failures, not retrieval — see below.

Hybrid search, reranking, recursive chunking, and contextual headers each
improved the *aggregate* number, but the single-hop/cross-reference gap has
stayed roughly the same size through all of them. The Step 9 investigation into
GDPR retrieval explains why: it's not that cross-reference questions are
"harder" in some vague sense — it's that they're phrased as plain-language
scenarios ("a CV-screening tool that processes... health-related disability
notes"), while the answer lives in formal statutory phrasing ("special
categories of personal data revealing... shall be prohibited unless..."). That
mismatch is deep enough that the truly relevant chunk can rank 65th-400th out
of 837 even at full-corpus depth — no amount of top-k tuning fixes that; it
needs query reformulation or a different retrieval strategy entirely.

## Other concrete gaps, ranked by how much they cost accuracy

1. **Citation accuracy is essentially fake.** Every judging round found
   citations mechanically echo whatever was retrieved rather than reflecting
   what was actually used (0.51-0.66 mean citation accuracy). For a compliance
   tool, this matters more than it might elsewhere — a user trusting the
   citations to check the law themselves is trusting a signal that isn't real
   yet.
2. **Generation-stage failures happen independently of retrieval quality.** The
   two worst single errors found across all judging rounds both had *good*
   retrieval: one question retrieved the exact answer at rank 1 and the model
   answered a different question entirely; another confidently cited Article
   54(6) instead of 53(2) — a real but wrong statutory citation — while a
   near-duplicate question elsewhere answered correctly. This
   self-inconsistency (same underlying fact, different answers depending on
   phrasing) is arguably the most concerning finding for a compliance-facing
   tool, since it's not something more retrieval tuning would fix.
3. **Doc-level retrieval metrics are unreliable in both directions.**
   Chunk-level checks (Step 8/9) found cases where the right document was
   retrieved but the wrong chunk within it (false positive), and cases where
   the automated check called it a miss but the fact was actually available in
   a different document entirely (false negative). The takeaway: our
   retrieval-hit numbers have real noise in them that a document-path check
   can't see.
4. **Cheap, mechanical corpus hygiene issues, not yet fixed:** Service Desk
   chunks carry ~100-150 tokens of navigation boilerplate; a GDPR "mirror"
   homepage with zero real content is ingested as 7 junk chunks; one specific
   paragraph (Article 50 guidelines, ¶38) seems to fall in a chunk-boundary gap
   and never gets retrieved by two different questions. None of these are hard
   to fix, they just haven't been prioritized yet.

## Bottom line

The pipeline is in good shape for single-document, directly-named questions
(~93% hit, ~95% correct) — that's most of a real compliance-assistant
workload. The remaining ~20-25 points of overall accuracy are concentrated in
a much smaller, structurally different problem: questions that require
combining two documents or mapping a scenario to the right legal concept.
That's not a retrieval-tuning problem in the "try another chunk size" sense
anymore — it needs either query-side intervention (reformulation/HyDE) or a
fundamentally different retrieval strategy for cross-reference questions
specifically. Citation accuracy and generation self-consistency are the two
next-biggest levers, and neither has been touched by any change so far —
they're pure generation/prompting problems sitting underneath whatever
retrieval improvements come next.
