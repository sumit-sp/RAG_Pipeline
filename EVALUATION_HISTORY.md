# Evaluation History

The single narrative of how this project's RAG pipeline was measured and improved,
step by step, with the actual eval numbers behind each change. For the *why* behind
each decision see `DECISIONS.md`; for the raw phase-by-phase task log see
`PROGRESS.md`; for the current-state results snapshot see `README.md`.

## 1. Overview

**Goal:** a RAG system answering compliance questions about the EU AI Act — e.g.
"which obligations apply to my AI system, and when" — built to demonstrate
evaluation discipline (every quality claim backed by a number, not a vibe), not
just to produce a working chatbot.

**Corpus** (`data/raw/`, manifest in `data/raw/SOURCES.md`): 8 source items —

- The AI Act itself (Regulation (EU) 2024/1689)
- The Digital Omnibus amendment (Regulation (EU) 2026/1744), which changed several
  dates written into the original AI Act — a genuine, checkable versioning test case
- GPAI Code of Practice (3 chapters: Transparency, Copyright, Safety & Security)
- Guidelines on the scope of GPAI obligations
- Guidelines on Article 50 transparency obligations
- Code of Practice on transparency of AI-generated content
- A curated set of AI Act Service Desk per-Article pages
- GDPR (Regulation (EU) 2016/679), for cross-regulation questions

**Golden set** (`eval/golden_set.jsonl`): 41 question/answer pairs. 6 were
hand-verified by reading the Digital Omnibus and original AI Act text directly
(the temporal-conflict cases); 37 were drafted by reading the rest of the corpus
and human-reviewed before being accepted — 2 additional drafted candidates were
rejected during that review and never made it in. Each question is tagged
`single-hop`, `multi-hop`, `cross-reference`, or `temporal-conflict`.

**How every number below was produced:** the live pipeline (real retrieval, real
generation) runs against all 41 questions, scored by DeepEval — a mix of
programmatic reference-based checks and LLM-judge checks (Groq `gpt-oss-120b`, a
bigger model than the `gpt-oss-20b` generator, used as judge). A run "passes" a
question only if every check for that run clears its threshold. **Read this before
trusting a single decimal place:** identical-config re-runs show real score
variance of ~0.05–0.07 per metric, because served LLM inference isn't bit-exact
even at `temperature=0`. Treat the overall pass count as the trustworthy signal.

## 2. Starting point: the naive baseline

**Architecture (Phase 1, "the dumbest complete pipeline"):**
- Parsing: BeautifulSoup for HTML, PyMuPDF for PDF — no per-document special-casing
- Chunking: fixed-size token windows, 500 tokens / 50 overlap, no structure-awareness
- Embeddings: local sentence-transformers (`BAAI/bge-small-en-v1.5`) — no API key
- Vector store: Qdrant, embedded/on-disk mode — no Docker required
- Retrieval: dense-only top-k, no hybrid, no reranking
- Generation: Groq `gpt-oss-20b`, grounded in retrieved context, with citations

**Eval harness (Phase 2):** DeepEval + pytest, scoring 4 pipeline-level metrics —
Faithfulness, Answer Relevancy, Contextual Precision, Contextual Recall.

**First baseline run:** 26/41 passed, and 2/41 questions crashed out of scoring
entirely with an empty Groq response — an unexplained gap at the time.

| Metric | Mean (n=39, excluding the 2 crashes) |
|---|---|
| Faithfulness | 0.952 |
| Answer Relevancy | 0.977 |
| Contextual Precision | 0.759 |
| Contextual Recall | 0.752 |

This number turned out to be unreliable (see Step 1) — it's kept here only as the
historical record of where the project actually started.

## 3. Step-by-step improvements

### Step 1 — Fix: empty Groq responses (not a quality change, a prerequisite)

While setting up the first real before/after comparison, root-caused the 2 crashed
questions: `gpt-oss-20b`/`gpt-oss-120b` are reasoning models that spend part of
their token budget on hidden "reasoning tokens" before the visible answer; on some
prompts this exhausted the default 2048-token budget entirely
(`finish_reason="length"`, empty content). Fixed with `max_completion_tokens=4096`
and `reasoning_effort="low"` on every Groq call.

Re-running dense-only after the fix, all 41 questions completed (0 crashes):

| Metric | Mean | Pass rate |
|---|---|---|
| Faithfulness | 0.887 | 37/41 |
| Answer Relevancy | 0.989 | 41/41 |
| Contextual Precision | 0.691 | 31/41 |
| Contextual Recall | 0.772 | 33/41 |
| **Overall (all 4 pass)** | | **27/41 (66%)** |

This — not the Phase 2 number above — is the real "before" every later change is
measured against.

### Step 2 — Hybrid search (dense + BM25 sparse) — KEPT

Added BM25 sparse vectors (fastembed's local `Qdrant/bm25` model) alongside the
existing dense embeddings, combined via Qdrant's native RRF fusion. No API key
needed.

| Metric | Dense (Step 1) | Hybrid | Δ |
|---|---|---|---|
| Faithfulness | 0.887 | 0.931 | +0.044 |
| Answer Relevancy | 0.989 | 0.917 | −0.072 |
| Contextual Precision | 0.691 | 0.754 | +0.063 |
| Contextual Recall | 0.772 | 0.776 | +0.004 |
| **Overall (all 4 pass)** | **27/41 (66%)** | **31/41 (76%)** | **+4 questions** |

**Why it helped:** dense embeddings match on semantic topic similarity, which can
miss exact-term hits (article numbers, defined terms) that BM25 catches directly —
precisely what multi-hop and cross-reference questions need. Multi-hop contextual
precision alone went 0.667 → 0.844. Kept as the new default.

### Step 3 — Cross-encoder reranking on top of hybrid — TRIED, REVERTED

Reranked hybrid's top-20 candidates with a local cross-encoder
(`cross-encoder/ms-marco-MiniLM-L-6-v2`), keeping the top-5.

| Metric | Hybrid (Step 2) | +Reranking | Δ |
|---|---|---|---|
| Faithfulness | 0.931 | 0.935 | +0.004 |
| Answer Relevancy | 0.917 | 0.984 | +0.067 |
| Contextual Precision | 0.754 | 0.718 | −0.036 |
| Contextual Recall | 0.776 | 0.809 | +0.033 |
| **Overall (all 4 pass)** | **31/41 (76%)** | **26/41 (63%)** | **−5 questions** |

**Why it hurt despite 2 of 4 average metrics improving:** the questions that got
worse (precision, and multi-hop specifically, which dropped from 0.844 to 0.602)
were different questions than the ones that got better — so the *set* of
questions passing all four checks shrank. `ms-marco-MiniLM-L-6-v2` is trained on
general web passage ranking, not legal text, and appears to fight hybrid search's
exact-term signal rather than refine it. **Reverted** — kept in the codebase, off
by default.

### Step 4 — Eval harness extended: component + application-level checks

Checked the eval suite itself against a standard component/pipeline/application
RAG-eval framework and found it only covered the pipeline level. Added:
- **Retrieval hit** (component/retriever level, programmatic, no LLM): does
  `expected_source_doc` actually appear in the top-k retrieved chunks?
- **Answer Correctness** (application level, DeepEval `GEval`): is the answer's
  content actually *correct* against the golden answer — not just consistent with
  whatever was retrieved?

| Check | Level | Result |
|---|---|---|
| Retrieval hit | Component | **33/41 (80%)** |
| Faithfulness | Pipeline | 0.951 mean, 39/41 |
| Answer Relevancy | Pipeline | 0.947 mean, 40/41 |
| Contextual Precision | Pipeline | 0.772 mean, 35/41 |
| Contextual Recall | Pipeline | 0.821 mean, 34/41 |
| Answer Correctness | Application | 0.751 mean, 33/41 |
| **All 6 checks pass simultaneously** | | **24/41 (59%)** |

**This 59% is not a quality regression from the 76% in Step 2** — it's the same
hybrid-search pipeline, measured more completely. The retrieval-hit check caught a
real, previously-invisible gap: on several questions, a *different* chunk than the
expected one contained similar-enough information to satisfy the LLM judge, even
though it wasn't the actual source the golden answer was written from. **24/41 is
the reference point for everything after this.**

### Step 5 — Chunk-size / chunking-strategy tuning (Groq-free screening)

Screened candidates using only the retrieval-hit check (deterministic, no LLM
calls at all) before spending any Groq calls on the full eval:

| Chunking | Retrieval hit rate |
|---|---|
| Fixed 500/50 (current) | 33/41 (80%) |
| Fixed 800/100 (larger) | 32/41 (78%) — flat/worse |
| **Recursive 500/50** (prefers paragraph/sentence/word boundaries over a hard token cut) | **34/41 (83%)** — best |

Recursive chunking is a small, real (non-noisy — this check has no LLM
variance), and so-far-unbeaten win. Not yet validated on the full 6-check,
Groq-based eval.

### Step 6 — External LLM-as-judge (Claude Sonnet 5) on hybrid + recursive chunking

Exported the live pipeline's output (hybrid retrieval + **recursive chunking** —
the untested-by-Groq winner from Step 5) against all 41 questions to
`eval/pipeline_outputs_for_external_judge.jsonl`, then had Claude Sonnet 5 judge it
independently of the Groq-family judge (`gpt-oss-120b` judging `gpt-oss-20b`
shares a lab and model family — an independent judge is a stronger check). Full
report: `eval/rag_eval_report.xlsx`.

**Note on comparability:** this run changed *two* things at once relative to the
Groq-based 24/41 reference point (Step 4) — the judge (Claude vs. Groq) *and* the
chunking strategy (recursive vs. fixed 500/50). The numbers below are not a clean
"Claude disagrees with Groq" comparison; a same-config, judge-only comparison
hasn't been run yet.

**Overall results (N=41):**

| Metric | Value | What it measures |
|---|---|---|
| Context Recall @5 (doc-level) | 82.9% (34/41) | Gold document present anywhere in top-5 retrieved chunks |
| Mean Reciprocal Rank (MRR) | 0.560 | How high the gold document ranks when found (1.0 = always rank 1) |
| Citation Recall | 82.9% (34/41) | Gold document appears in the answer's citation list |
| Faithfulness (judge-scored) | 0.959 | Every claim in the answer traces back to retrieved context |
| Answer Correctness (judge-scored) | 0.780 | Answer matches the gold answer in substance, including carve-outs |
| Answer Relevancy | 1.00 (41/41) | All answers stay on-topic |

The 34/41 (82.9%) doc-level recall figure is an **exact match** with the
Groq-free retrieval-hit screen for recursive chunking in Step 5 (also 34/41) —
independent confirmation that recursive chunking's retrieval win is real, from a
completely different measurement method.

**By difficulty:**

| Difficulty | n | Retrieval Hit@5 | MRR | Faithfulness | Answer Correctness |
|---|---|---|---|---|---|
| single-hop | 28 | 92.9% | 0.671 | 1.00 | 0.857 |
| cross-reference | 8 | 62.5% | 0.354 | 0.850 | 0.744 |
| multi-hop | 3 | 33.3% | 0.167 | 0.833 | 0.433 |
| temporal-conflict | 2 | 100% | 0.417 | 1.00 | 0.375 |

Same story as every earlier measurement: single-hop is solid, multi-hop and
cross-reference are materially weaker (correctness ~0.43-0.74 vs. ~0.86).

**Outcome distribution:** Correct 27, Correct-Partial 1, Correct-Condensed 1
(→ 29/41, 71% fully correct or correct-with-minor-gaps) · Partial 6,
Partial/Incorrect 1 (7/41, 17%) · Abstained-Justified/Task Fail 4 (10%, the model
correctly declined rather than guess) · Incorrect/Hallucinated 1 (2%).

**Key findings — new, concrete, and actionable:**

1. **Citations aren't an independent signal.** In 41/41 records, the `citations`
   list is exactly the set of unique document paths in the retrieved chunks — the
   pipeline echoes back whatever was retrieved rather than selecting which sources
   actually support the answer. Citation Recall is therefore mechanically
   identical to retrieval hit-rate, not a real check on generation quality.
2. **Document-level retrieval hit-rate overstates true recall.** Two verified
   cases retrieved the *correct document* but the *wrong chunk within it* — the
   specific sentence needed (a compliance date, a 5-business-day SLA) sat in a
   different chunk that wasn't fetched. Both count as "hits" at the document
   level but the task still failed. This is the single largest driver of the gap
   between an 83% document-level hit-rate and a 78% answer-correctness rate.
3. **One clear hallucination under good retrieval.** The gold answer (GDPR Art.
   9's "without prejudice" clause) was present verbatim in two retrieved chunks,
   yet the model built a fluent, confident answer around the wrong regulation
   (misnaming Directive 2016/680 as the "Biometric Data Directive") and never
   mentioned GDPR Art. 9 — a plausible-sounding but substantively wrong answer to
   a legal cross-reference question, the most consequential single error found.
4. **Self-inconsistency across near-duplicate questions.** Asked essentially the
   same fact two ways, the system correctly cited Art. 53(2) for the open-source
   exemption in one answer, then claimed in the next that the exemption is "set
   out in Article 53(1)(a) and (b) itself" — contradicting its own prior answer.
   No self-consistency/verification step exists across paraphrased queries.
5. **Multi-hop and cross-reference remain the weak point**, confirming every
   earlier measurement in this document from an independent judge and a changed
   chunking strategy.
6. **Completeness lags even on retrieval "hits."** Several records state the
   correct top-level answer but drop a specific carve-out, exception, or
   safeguard the gold answer treats as essential (e.g. GDPR Art. 22(2)/(3)
   safeguards, a law-enforcement exception). Retrieval succeeded; the gap is in
   extraction/synthesis completeness.
7. **Chunking carries boilerplate noise.** Many Service Desk article chunks are
   prefixed with ~100-150 tokens of repeated navigational text (a full table of
   contents) before the substantive content, diluting effective context
   precision even in chunks that are otherwise correctly ranked and on-topic.

### Step 7 — Contextual chunk headers (Groq-free screening) — KEPT, new best

Per Anthropic's "contextual retrieval" technique, and per the new standing rule
that LLM-content-generation tasks go through an external Sonnet-5 export rather
than Groq: exported every source document's full text plus its fixed-500/50
chunk list to `eval/chunks_for_contextual_headers.jsonl` (25 docs, 837 chunks —
`eval/export_chunks_for_contextual_headers.py`), had the user run it through
Claude Sonnet 5 externally, and got back `eval/contextual_headers.jsonl` (837
one- to two-sentence headers, one per chunk, situating each chunk within its
document before embedding — see `eval/CONTEXTUAL_HEADERS_INSTRUCTIONS.md`).

Re-ingested with `USE_CONTEXTUAL_HEADERS=true` (fixed 500/50 chunking + hybrid
retrieval — the current default combination, matched to the chunking used when
the headers were generated so the `(source_doc, chunk_index)` mapping stays
valid) and re-ran the Groq-free retrieval-hit screen:

| Chunking / config | Retrieval hit rate |
|---|---|
| Fixed 500/50 + hybrid (baseline) | 33/41 (80%) |
| Recursive 500/50 + hybrid (Step 5) | 34/41 (83%) |
| **Fixed 500/50 + hybrid + contextual headers** | **36/41 (87.8%)** — new best |

Contextual headers beat recursive chunking's win from Step 5, using the same
fixed-chunking base it was generated against. Remaining misses: one
Digital-Omnibus-amended-date question, an Article 53 open-source-exemption
question, two special-category-personal-data-in-recruitment questions, and one
Article 50(1)/GPAI-transparency-overlap question — all cross-reference or
multi-hop by nature, consistent with every earlier measurement in this
document.

**Not yet tried:** contextual headers combined with recursive chunking (would
need regenerating the header export against recursive chunk boundaries first,
since headers are keyed by chunk index and the two chunkers produce different
boundaries) — parked at the user's request until remaining tasks are done.

### Step 8 — External LLM-as-judge (Claude Sonnet 5) on a targeted 15-question hard subset

Rather than spend a full 41-question judge pass on the contextual-headers
pipeline, judged a **targeted hard subset**: every cross-reference (8),
multi-hop (3), and temporal-conflict (2) question — 13 structurally hard by
the golden set's own label — plus the 2 single-hop questions that still miss
the Groq-free retrieval-hit check (Step 7) — **15 questions**. This follows
the spec's Tier 0/1/2 cost cascade: cheap/free checks first, expensive judging
reserved for what's still uncertain, rather than re-judging the ~27
already-easy single-hop questions every prior full-set run confirms are fine.

Mechanics: `eval/export_hard_subset_for_external_judge.py` ran the live
pipeline (hybrid retrieval + contextual headers, one Groq generation call per
question) and wrote `eval/hard_subset_outputs_for_external_judge.jsonl`.
Claude Sonnet 5 judged it externally per `eval/HARD_SUBSET_JUDGE_INSTRUCTIONS.md`,
returning `eval/hard_subset_judge_scores.jsonl` — the rubric adds two metrics
to Step 6's set, both aimed at Step 6's own open findings: `chunk_level_hit`
(does a retrieved chunk contain the actual needed fact, not just the right
document?) and `citation_accuracy` (do citations reflect real support, or just
echo the retrieved set?).

**Overall results (N=15, the hard subset only — not comparable to full-set
percentages above):**

| Metric | Value |
|---|---|
| Retrieval Hit (doc-level) | 10/15 (67%) |
| Chunk-Level Hit | 10/15 (67%) — but a *different* 10 (see below) |
| MRR | 0.550 |
| Faithfulness | 0.803 |
| Answer Correctness | 0.600 |
| Citation Accuracy | 0.513 |
| Outcome distribution | Correct 2, Correct-Partial 5, Correct-Condensed 1 (→ 8/15, 53% fully/mostly correct) · Partial 4 · Task-Fail 1 · Abstained-Justified 1 · Incorrect 1 |

Lower than Step 6's full-set 71%/83% figures **by design** — this is the
hardest 15 questions in the golden set, not a regression.

**Key findings — new, concrete, and actionable:**

1. **Doc-level retrieval-hit both over- and under-counts correctness, in
   opposite directions, on the same 15 questions.** 2 cases (of 10 doc-level
   "hits") retrieved the right document but the wrong chunk — confirming Step
   6's finding 2 again. But separately, 2 cases the automated check counts as
   *misses* actually had the needed fact available anyway, verbatim, in a
   *different* retrieved document (e.g. the Digital Omnibus's recital text
   quotes the very AI Act date it supersedes) — a **false negative** in
   doc-path-based scoring that Step 6 didn't surface. Net: document-path
   matching is noisier in both directions than a single "hit rate" number
   suggests.
2. **GDPR cross-reference retrieval is inconsistent, not uniformly weak.** Of
   6 GDPR cross-reference questions in this subset, 3 retrieved zero GDPR
   chunks at all (CV-screening/health-data, special-category recruitment bias,
   chatbot Art.50/GDPR-notice), while the other 3 retrieved GDPR cleanly and
   scored well (Art.22 automated-decision rights, Art.35 DPIA, Art.9 biometric
   data). The pattern: GDPR retrieval succeeds when the question names a
   specific mechanism (DPIA, automated decision-making) but fails when the
   question is a scenario needing the model to infer *which* GDPR concept
   applies — a harder retrieval-relevance problem than simple term matching.
3. **A confidently wrong statutory citation, self-inconsistent with an
   adjacent question.** Asked where the Article 53(1)(a)/(b) open-source
   exemption is written, one answer confidently cites Article 54(6) (a real,
   verbatim, but *substantively different* provision — the authorised-
   representative exemption, not the open-source one) as if it matches the
   Service Desk's description. A near-duplicate question two rows later
   correctly identifies Article 53(2) instead and correctly distinguishes it
   from Article 54(6) — the same self-inconsistency pattern Step 6 found,
   this time with a wrong legal citation as the consequence, the single most
   serious error in this subset for a compliance-facing tool.
4. **Citation accuracy remains low (0.513 mean) and is the most consistently
   weak metric** — nearly every record's comment notes citations listing
   documents that weren't actually used, confirming Step 6's finding 1 persists
   under contextual headers.
5. **Generation-stage failures happen even with perfect retrieval.** One
   question retrieved the exact answer at rank 1, verbatim, yet the generated
   answer addressed a different, unrelated question entirely using lower-ranked
   chunks — a pure generation-stage miss, not a retrieval problem, and the
   lowest answer-correctness score in the subset (0.05) despite `mrr=1.0`.
6. **Boilerplate/navigation noise still wastes retrieval slots** — one Service
   Desk chunk retrieved for a question was pure site-navigation text with no
   substantive content, confirming Step 6's finding 7 is still present after
   contextual headers were added (headers describe the boilerplate-heavy
   chunk accurately, but don't remove the boilerplate itself).

## 4. Pass-rate timeline at a glance

| Stage | Checks used | Overall pass rate |
|---|---|---|
| Naive baseline (dense, first run, 2 crashes) | 4 pipeline | 26/41 (2 crashed) |
| Naive baseline (dense, bug-fixed) | 4 pipeline | 27/41 (66%) |
| + Hybrid search | 4 pipeline | 31/41 (76%) |
| + Reranking (reverted) | 4 pipeline | 26/41 (63%) |
| + Retrieval-hit & Answer-Correctness checks | 6 total | **24/41 (59%) — current reference point** |
| + Recursive chunking (Groq-free screen only) | retrieval-hit only | 34/41 hit rate (83%) |
| + Recursive chunking, judged by Claude Sonnet 5 | independent judge, 6 metrics | 34/41 doc-recall (83%, confirms the screen); 29/41 (71%) fully/mostly correct by outcome label |
| + Contextual chunk headers (fixed chunking + hybrid) | retrieval-hit only | **36/41 hit rate (87.8%) — new best** |
| + Contextual headers, judged by Claude Sonnet 5 on hard 15-question subset | independent judge, 6 metrics, hardest questions only | 10/15 doc-recall (67%); 8/15 (53%) fully/mostly correct by outcome label — not comparable to full-set % above |

## 5. What's still open

- **Run recursive chunking through the same Groq-based 6-check harness** used for
  Step 4, so it can be directly compared to the 24/41 reference point on equal
  footing (Step 6 changed the judge and the chunking strategy at once).
- **Implement claim-level citation attribution** (Finding 1) — citations
  currently just echo the retrieved set rather than reflecting what actually
  supports the answer.
- **Add a chunk-level recall metric**, not just document-level (Finding 2,
  reconfirmed in Step 8 finding 1) — the biggest measured driver of the gap
  between retrieval-hit-rate and answer-correctness. Step 8 also found the
  reverse failure mode (doc-path scoring false negatives), so this metric
  needs to handle both directions, not just tighten the existing one.
- **Strip navigation/boilerplate at ingestion** for Service Desk article chunks
  (Finding 7, reconfirmed in Step 8 finding 6) — cheap, mechanical, and
  currently diluting context precision even with contextual headers on.
- Consider a self-consistency check across paraphrased queries (Finding 4,
  reconfirmed sharply in Step 8 finding 3 — this time producing a wrong
  statutory citation, the most consequential error found so far).
- **Investigate inconsistent GDPR cross-reference retrieval** (Step 8 finding
  2, new) — succeeds for named mechanisms (DPIA, Art. 22), fails for
  scenario-phrased questions needing inference to the right GDPR concept.
- Contextual headers judged on the full 41-question 6-check harness — Step 8
  intentionally judged only the 15-question hard subset (cheaper, targeted);
  the easier ~26 single-hop questions are not yet re-judged under headers,
  though the Groq-free screen (Step 7) covers all 41.
- Contextual headers + recursive chunking stacked together — parked at the
  user's request until remaining tasks are complete.
- Phase 4 (ingestion sophistication — effective-date metadata, cross-reference
  linking) not started; the retrieval-hit misses (Digital-Omnibus-adjacent and
  cross-reference questions) remain the concrete targets, now with two
  independent judges' worth of evidence pointing at the same weak spots.
- Deployment (Render/Railway) deferred pending a separate GitHub account setup.
