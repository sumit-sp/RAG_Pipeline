# Eval results

Baseline numbers for the Phase 1 walking skeleton (plain backend): naive HTML/PDF
parsing, fixed-size chunking, local embeddings, dense-only top-k retrieval, Groq
`gpt-oss-20b` generation. Measured against all 41 questions in `eval/golden_set.jsonl`
via DeepEval, judged by Groq `gpt-oss-120b` (see `DECISIONS.md` for the judge-model
trade-off). Metric threshold: 0.5 for all four metrics. Raw per-question scores are
in `eval/eval_run_raw_results.jsonl`.

**Overall test result: 26/41 questions (63%) passed on all four metrics simultaneously.**
2 of 41 questions errored before scoring (empty generation output from Groq on both —
see "Known issues" below) and are excluded from the metric averages, which are
computed over the 39 questions that completed.

## Baseline scores (n=39)

| Metric | Threshold | Mean score | Pass rate |
|---|---|---|---|
| Faithfulness | 0.5 | **0.952** | 38/39 (97%) |
| Answer Relevancy | 0.5 | **0.977** | 39/39 (100%) |
| Contextual Precision | 0.5 | **0.759** | 32/39 (82%) |
| Contextual Recall | 0.5 | **0.752** | 34/39 (87%) |

## By difficulty

| Difficulty | n | Faithfulness | Answer Relevancy | Contextual Precision | Contextual Recall |
|---|---|---|---|---|---|
| single-hop | 26 | 0.946 | 0.974 | 0.823 | 0.788 |
| cross-reference | 8 | 0.943 | 0.972 | **0.626** | **0.542** |
| multi-hop | 3 | 1.000 | 1.000 | 0.667 | 0.833 |
| temporal-conflict | 2 | 1.000 | 1.000 | 0.600 | 1.000 |

## Reading the baseline

- **Generation is already strong.** Faithfulness (0.95) and answer relevancy (0.98) are
  high across the board — when the pipeline retrieves *something* relevant, `gpt-oss-20b`
  answers faithfully to it and stays on-topic. This includes both temporal-conflict
  questions (the Digital-Omnibus-vs-original-Article-113 dates), which scored perfectly
  on faithfulness/relevancy, consistent with the manual spot-check in `PROGRESS.md`.
- **Retrieval is the bottleneck, and it's worst on cross-reference questions.**
  Contextual precision/recall are meaningfully lower than faithfulness/relevancy overall
  (0.76/0.75 vs. 0.95/0.98), and cross-reference questions — the ones requiring chunks
  from two different documents (e.g. a Service Desk article's summary vs. the AI Act's
  primary text, or an AI Act obligation vs. its GDPR counterpart) — score worst of all
  (0.63 precision, 0.54 recall). This is exactly what you'd expect from dense-only
  top-k search with no query decomposition or multi-document awareness: a single
  embedding of the question tends to pull chunks from whichever one document is the
  closest semantic match, not both documents a cross-reference question actually needs.
  **This is the clearest, most specific target for Phase 3** (hybrid search and/or
  reranking should be evaluated against cross-reference questions specifically, not
  just the overall average).
- Sample size caveat: 39 questions (and subsets as small as 2-8 per difficulty bucket)
  is enough to see a real, actionable signal here, but not enough to treat any single
  decimal place as precise — Phase 3 comparisons should look for consistent directional
  movement, not chase noise in the third digit.

## Known issues surfaced by this baseline run

- **2 of 41 questions (both about the GPAI Code of Practice's Safety and Security
  chapter) got an empty generation response from Groq** and errored out of DeepEval's
  metrics entirely (`MissingTestCaseParamsError: 'actual_output' cannot be empty`)
  rather than producing a low score. Not yet root-caused — candidates are a transient
  Groq API issue or something specific to those two prompts/retrieved contexts (both
  draw from the same PDF). Worth a manual retry and, if it recurs, hardening
  `PlainGenerator.generate()` to handle an empty/`None` response explicitly rather than
  silently passing it through. Left as a Phase 3+ item since Phase 2's job is measuring,
  not fixing.
- No reranking, no hybrid search, no query rewriting — all as specced for the Phase 1
  baseline. The contextual precision/recall numbers above are what that gets you on
  this corpus.
