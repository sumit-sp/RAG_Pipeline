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

**A second, smaller noise source found later (Step 14 follow-up, migrating to
Qdrant Cloud):** retrieval itself can also be non-deterministic at the margin,
independent of any LLM. Repeated identical queries against the same Qdrant
collection occasionally returned a different chunk in the last retrieved slot.
Root cause: RRF (reciprocal rank fusion) scores are coarse, quantized values
(1.0, 0.5, 0.333, 0.25, ...), so exact ties at the top-k cutoff boundary are
common, and which tied chunk wins isn't always stable across calls. Confirmed
this is not a local-vs-cloud difference — 3 repeated calls against the same
cloud cluster for the same question gave 2 different outcomes. Measured
impact: comparing `eval/compute_retrieval_metrics.py` on local vs. Qdrant
Cloud gave near-identical aggregates (Precision 0.295 vs. 0.298, Recall 0.698
vs. 0.707) with exactly one question's hit count differing by one chunk. Small,
but worth knowing before reading a single question's Precision/Recall as
exact.

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

### Step 9 — Investigation: why is GDPR cross-reference retrieval inconsistent?

Step 8 found GDPR cross-reference retrieval succeeding on 3 of 6 questions and
failing entirely on the other 3, without explaining why. Investigated with
`eval/diagnose_gdpr_retrieval.py` — no LLM calls, pure retrieval-mechanics
inspection — by checking dense-only, sparse-only, and hybrid-RRF rank
positions for the *real* GDPR document (`adjacent/gdpr_2016_679.html`) at
top-20 (4x past the final top-5 cutoff), and for two of the six questions,
across the *entire* 837-chunk corpus to see how far off the truly relevant
chunk actually ranks.

**Three distinct root causes found, not one:**

1. **"Just below the final cutoff"** (2 of 6 questions — CV-screening,
   special-category recruitment bias). The relevant GDPR chunk *is* found by
   sparse (and sometimes dense) within the top-20 wide candidate pool — ranks
   6-14 — but RRF fusion (which penalizes a chunk that one of the two signals
   ranks very low or misses entirely) pushes the combined rank to 9-19,
   outside the final `RETRIEVAL_TOP_K=5` cutoff. This is a genuine, fixable
   "too aggressive a cutoff for cross-reference questions" problem.
2. **Deep semantic/lexical mismatch, not a cutoff problem** (the same 2
   questions, checked further). Extending the same two queries to rank
   against the *full* 837-chunk corpus (not just the top-20 window) found that
   the actual best-matching chunk — GDPR Article 9's operative "special
   categories of personal data" text — ranks 65-418th by dense similarity and
   22-298th by sparse/BM25. What *did* surface in the top-20 (GDPR recitals on
   a tangentially related topic) were lucky partial matches, not the real
   answer. Scenario-phrased business questions ("a CV-screening tool that
   processes... health-related disability accommodation notes") apparently
   embed poorly against formal statutory phrasing ("special categories of
   personal data revealing... shall be prohibited... unless..."), even at
   very generous recall depth. This is a genuine retrieval-relevance gap, not
   something a bigger top-k or better fusion weighting fixes.
3. **Compound dual-regulation queries get dominated by whichever regulation's
   vocabulary is more prominent** (1 of 6 questions — the Article 50(1)
   chatbot/GDPR-notice question). The query is phrased mostly in AI-Act
   transparency terms ("Article 50(1)", "chatbot", "you are talking to an
   AI"); the GDPR Art. 13/14 chunk that actually answers it ranks 293rd-336th
   out of 837 by dense similarity — invisible even at very wide recall. Every
   one of the top-20 hits (both dense and sparse) is instead from the
   Article-50 transparency-guidelines PDF, which shares surface vocabulary
   with the query. Contextual headers don't help here: each document's
   headers were generated by Sonnet-5 seeing only that document's own text in
   isolation, so a GDPR Art. 13/14 header describes GDPR concepts only, with
   no hint that the chunk is also relevant to AI Act transparency compliance —
   headers can't bridge a cross-document relationship neither side's header
   text expresses.

**Where retrieval succeeds** (3 of 6 questions — automated-decision rights,
DPIA, biometric data), the query phrasing closely mirrors the actual legal
mechanism's name ("automated decision", "DPIA", "biometric data") — both
dense and sparse agree strongly and rank the correct chunk 1st-3rd, so hybrid
fusion keeps it comfortably inside the top-5. **The dividing line isn't GDPR
vs. AI-Act, or cross-reference vs. single-hop — it's whether the question
names the specific legal mechanism it's asking about, or only describes a
scenario the model has to map to one.**

**One unrelated but real finding surfaced along the way:**
`adjacent/gdpr_2016_679_mirror.html` — confirmed in `data/raw/SOURCES.md` to
be a pure site-navigation homepage with no actual article text — is being
ingested as 7 real chunks (of 837 total) because the ingestion skip filter
only matches filenames containing "index", not "mirror". Low blast radius
(7/837 chunks) but the same class of problem as the already-flagged Service
Desk boilerplate finding (Step 6/8) — cheap to fix by extending the skip
filter, not yet done.

**Not a real problem, despite zero GDPR retrieval:** the Article 5(1)(h)
biometric-data question also retrieves zero GDPR chunks at any rank, but
still scored well in Step 8, because the exact needed cross-reference clause
("without prejudice to Article 9 GDPR... for purposes other than law
enforcement") happens to be duplicated verbatim inside the AI Act's own
regulation text — corpus redundancy compensating for a genuine retrieval
miss, not evidence the miss doesn't exist.

**Implications, not yet acted on:** raising `RETRIEVAL_TOP_K` would fix
failure mode 1 only; it would not fix modes 2 or 3, which need either query
reformulation (rewriting scenario questions toward statutory phrasing before
embedding, or HyDE-style hypothetical-document embedding) or a
metadata-filtered secondary retrieval pass specifically over the
adjacent/GDPR sub-collection for cross-reference questions. No config change
made yet — this step is diagnostic only.

### Step 10 — External LLM-as-judge (Claude Sonnet 5) on the full 41-question set

Extended Step 8's judging from the 15-question hard subset to **all 41**
questions, so the full picture is directly comparable rather than
deliberately skewed toward the hardest tail. Mechanics:
`eval/export_contextual_headers_full_for_external_judge.py` ran the live
pipeline (hybrid retrieval + contextual headers) for all 41 questions and
wrote `eval/contextual_headers_full_outputs_for_external_judge.jsonl`; Claude
Sonnet 5 judged it per `eval/CONTEXTUAL_HEADERS_FULL_JUDGE_INSTRUCTIONS.md`
(same 6-metric rubric as Step 8), returning
`eval/contextual_headers_full_judge_scores.jsonl`. For the 15 questions
already judged in Step 8, the judge reproduced its earlier scores rather than
re-litigating them (regenerated answers differed only in wording, not
substance) — so Step 10's numbers are a strict superset, not a re-roll.

**Overall results (N=41):**

| Metric | Value |
|---|---|
| Retrieval Hit (doc-level) | 36/41 (87.8%) — **exact match** with the Groq-free screen (Step 7) |
| Chunk-Level Hit | 34/41 (82.9%) |
| MRR | 0.715 |
| Faithfulness | 0.909 |
| Answer Correctness | 0.828 |
| Citation Accuracy | 0.663 |
| Outcome distribution | Correct 23, Correct-Partial 10, Correct-Condensed 1 (→ 34/41, 82.9% fully/mostly correct) · Partial 4 · Task-Fail 1 · Abstained-Justified 1 · Incorrect 1 |

The 36/41 doc-level figure being an exact match with the Groq-free
retrieval-hit screen (Step 7) is the same kind of independent cross-check
Step 6 found for recursive chunking — two different measurement methods
agreeing gives real confidence contextual headers' retrieval win is genuine,
not a screening artifact.

82.9% fully/mostly correct is the best outcome-distribution result of any
full-set judged run so far (Step 6's recursive-chunking run was 71%).

**By difficulty:**

| Difficulty | n | Retrieval Hit | Chunk-Level Hit | Faithfulness | Answer Correctness |
|---|---|---|---|---|---|
| single-hop | 28 | 26/28 (93%) | 26/28 (93%) | 0.971 | 0.950 |
| cross-reference | 8 | 6/8 (75%) | 6/8 (75%) | 0.769 | 0.669 |
| multi-hop | 3 | 2/3 (67%) | 1/3 (33%) | 0.683 | 0.633 |
| temporal-conflict | 2 | 2/2 (100%) | 1/2 (50%) | 0.925 | 0.050 |

Single-hop is now excellent across the board (0.950 answer correctness) —
**confirming the Step 8 targeting logic was sound**: the 26 additional
single-hop questions judged here for the first time hold up just as well as
the retrieval-hit screen implied, so the hard-subset strategy of
concentrating judging effort on cross-reference/multi-hop/temporal-conflict
questions wasn't hiding a hidden single-hop problem.

Temporal-conflict's 0.050 answer-correctness despite 100% retrieval hit and
0.925 faithfulness is **not a retrieval problem** — it's the same two
generation-stage failures already found in Step 8 (one pure non-responsive
answer despite perfect top-1 retrieval; one correct, justified abstention
that still scores 0 on a strict correctness metric). Multi-hop and
cross-reference remain the structurally weak categories, consistent with
every prior measurement in this document.

**One new finding from the 26 previously-unjudged questions:** two different
Article 50 transparency-guidelines questions (about terms-and-conditions-only
disclosure sufficiency, and machine-generated-text labeling) both miss the
same specific passage — paragraph 38 of the transparency guidelines — with
retrieved chunks covering paragraphs 34-37 but stopping just short of it. The
judge flagged this as "a systematic chunking/retrieval miss for that specific
passage rather than a one-off," since both answers reach the right
conclusion by extrapolating from nearby-but-not-identical text rather than
citing the actual controlling passage — a concrete, fixable chunk-boundary
issue for that document, not yet investigated further.

### Step 11 — GDPR cross-reference boost — KEPT, new best

Acted on the Step 9 diagnosis. Root cause, in plain terms: GDPR is only ~150
of the corpus's 837 chunks, and the normal search ranks every chunk against
the whole corpus at once — so for a question phrased mostly in AI-Act
vocabulary, the AI-Act-heavy majority of the corpus can out-rank a GDPR chunk
even when that GDPR chunk is the right answer. It isn't that the content is
missing or the search is broken; GDPR is just too small a minority to win a
corpus-wide vocabulary contest.

**Fix implemented** (`app/pipelines/plain/retrieval.py`,
`USE_CROSS_REFERENCE_BOOST`, default on): when a question contains the word
"GDPR" (case-insensitive), run one additional search restricted to just
`adjacent/gdpr_2016_679.html` (via a Qdrant metadata filter — no other
document competing), and append any new chunks it finds (up to
`CROSS_REFERENCE_BOOST_LIMIT`, default 5) to the normal top-k. No LLM call
involved, so it's unaffected by the Groq-restriction rule, and it's fully
deterministic and cheap (one extra Qdrant query, only for GDPR-mentioning
questions).

**Validated before implementing:** re-ran the 3 previously-failing questions
with a GDPR-only filtered search and confirmed the actual needed chunk (GDPR
Article 9's operative text for 2 of them, Article 13/14 for the third) now
ranks 1st-5th within that restricted search — up from being invisible
(rank 65th-336th) against the full 837-chunk corpus.

**Result — Groq-free retrieval-hit screen:**

| Config | Retrieval hit rate |
|---|---|
| Contextual headers, no boost (Step 7) | 36/41 (87.8%) |
| **Contextual headers + GDPR cross-reference boost** | **39/41 (95.1%)** — new best |

All 3 previously-failing GDPR cross-reference questions now pass, with **zero
regressions** elsewhere. The 2 remaining misses are the same doc-path-scoring
false negatives already documented in Step 8/9 (the fact is actually present
in a different retrieved document) — not new failures.

**Scope and limits, noted honestly:** this is a targeted, keyword-triggered
fix for the one diagnosed gap (GDPR), not a general solution to cross-document
retrieval. It works here because every GDPR-cross-reference question in the
golden set happens to say "GDPR" explicitly — a real system facing more
varied phrasing (e.g. a question that means GDPR without naming it) would
need the query-reformulation or routing approaches Step 9 also considered.
Also not yet re-validated with a full Sonnet-5 judge pass — only the
Groq-free retrieval-hit screen — so the downstream effect on answer
correctness (not just retrieval) isn't measured yet.

### Step 12 — Corpus-side cross-reference tagging — KEPT, new best, generalizes Step 11

Step 11's fix only helped when the *question* said "GDPR." The user asked
whether that would generalize to other documents or to phrasing that implies
a cross-referenced regulation without naming it — it wouldn't have. This step
replaces the question-only signal with a second, corpus-driven one.

**What changed:** at ingestion, every chunk is now scanned for explicit
mentions of other documents in the corpus (a small alias registry —
`app/pipelines/plain/cross_references.py` — maps distinctive phrases like
"Regulation (EU) 2016/679" or "Digital Omnibus" to the document they name)
and tagged with a `references` list. At retrieval time, the boost now fires
on **either** signal: the question naming a document (Step 11, kept as a
cheap first check) **or** an already-retrieved chunk's own `references` tag
naming a document not yet represented (new — reacts to what the *documents*
say, not how the *question* is phrased). Both signals share the same alias
registry, so there's one source of truth. Registration is deliberately
conservative — only documents with a distinctive, unambiguous phrase (an
official regulation number) are registered; the three GPAI Code of Practice
chapters, for example, share a generic name and aren't, since a mention of
"GPAI Code of Practice" alone can't say which of the three PDFs is meant.

**Result — Groq-free retrieval-hit screen:**

| Config | Retrieval hit rate |
|---|---|
| Contextual headers + GDPR keyword boost only (Step 11) | 39/41 (95.1%) |
| **+ corpus-side cross-reference tagging** | **40/41 (97.6%)** — new best |

The remaining miss is the already-documented Article 53 open-source-exemption
question, itself a doc-path false negative (Step 8) — the content is present
in the primary regulation and GPAI scope guidelines, just not literally in
the nominal "expected" Service Desk page.

**Confirmed it actually generalizes, not just re-derives Step 11:** a second,
previously-failing question — the original-application-date
temporal-conflict question, a Step 8/9 doc-path false negative — also now
passes, unprompted: a top-ranked Digital Omnibus chunk explicitly names
"Regulation (EU) 2024/1689," which the corpus-side signal picked up on its
own, pulling in the actual AI Act Article 113 text as a side effect. This is
a document pair (AI Act ↔ Digital Omnibus) the fix wasn't targeted at — real
evidence the mechanism generalizes beyond the one diagnosed GDPR case.

**Honest limit, verified directly:** the corpus-side signal only fires when a
*retrieved* chunk explicitly names the other document — it doesn't detect
topical relevance on its own. Testing a deliberately rephrased version of one
of the GDPR questions that never says "GDPR" anywhere ("If an AI tool used
for hiring processes candidates' health information, what extra data
protection obligations apply beyond the AI Act itself?") confirmed the boost
correctly does *not* fire for it, because none of its top-5 retrieved chunks
happen to name GDPR in their own text either. So this is a broader,
phrasing-independent version of "detect an explicit named reference," not a
general "detect topical relevance" solution — a question whose
best-matching chunks never happen to name the other document in so many
words remains unsolved by this mechanism, and would need query reformulation
or a semantic router instead.

**Scope check — how often does this actually fire?** 15 of 41 golden-set
questions now trigger a boost (up from ~6-8 under the keyword-only version),
averaging ~5 extra chunks each. At the time this step was written, these
firings had only been spot-checked for whether the *topic* looked plausibly
related (e.g. AI Act Articles 10/11/12/14 do explicitly invoke GDPR
somewhere in their own text) — **this held up less well than it looked**:
Step 13's actual Sonnet-5 judging of these 15 questions found that
topic-level plausibility isn't the same as question-level relevance, with
real costs (a systematic citation-accuracy drop, and two real answer-quality
regressions) alongside real wins. See Step 13 for the full, judged picture.

### Step 13 — Judged the cross-reference boost's 15 affected questions: a real, mixed result

Step 12's own "spot-checked and all firings look legitimate" claim doesn't
survive Sonnet-5 judging — topical relevance (the AI Act article *does*
discuss something GDPR-adjacent) turned out not to mean the specific
boosted-in GDPR chunk was actually relevant to *that question*. Judged only
the 15 questions the boost touches (`eval/boosted_subset_judge_scores.jsonl`),
compared directly against each question's existing Step 10 score (before the
boost existed) — a clean, matched before/after on the same 15 questions:

| Metric | Before (Step 10) | After (boost fires) | Δ |
|---|---|---|---|
| Retrieval Hit | 73.3% | 100% | **+26.7 pts** |
| Chunk-Level Hit | 66.7% | 93.3% | **+26.7 pts** |
| MRR | 0.633 | 0.668 | +0.035 |
| Faithfulness | 0.843 | 0.867 | +0.023 |
| Answer Correctness | 0.803 | 0.760 | **−0.043** |
| Citation Accuracy | 0.617 | 0.510 | **−0.107** |

Retrieval got unambiguously better. **Answer correctness and citation
accuracy both got slightly worse on average**, and the aggregate hides a
sharper split underneath:

**Real, substantial wins (the mechanism doing exactly what it was built
for) — 4 questions:** the CV-screening GDPR question jumped 0.35 → 0.65
(Correct-Partial vs. previously Partial) as GDPR special-category rules
were finally grounded instead of hand-waved; the recruitment-bias GDPR
question and the Article-50-watermark question both improved on
faithfulness/citation-accuracy from newly-surfaced, genuinely relevant text;
and the GPAI open-source-exemption question gained a second, independent,
correct confirmation of the Article 53(2)/54(6) distinction — the exact
distinction whose confusion caused the worst error found in Step 8.

**One severe regression — the single most concerning result from this
round:** "Does the Digital Omnibus change when Chapters I and II of the AI
Act apply?" went from a correct, if incomplete, answer (0.55, Partial) to a
**non-answer** (0.10, Task-Fail). The boost appended 5 GDPR chunks with zero
bearing on the question (adequacy decisions, processor contracts, delegated
acts); the model, faced with that irrelevant bulk alongside the actually-useful
chunk it already had, second-guessed itself into "I cannot confirm" instead
of using the evidence it had. **More context made the answer worse, not just
unhelpful.**

**One moderate regression:** the Article 5(1)(h)/GDPR-biometric-interaction
question dropped from 0.55 to 0.45 and produced a longer,
table-formatted answer that **contradicts itself** — one row states GDPR's
Article 9(1) "still applies" to the law-enforcement case the AI Act exempts,
which is backwards, while the same answer's own bottom line gets the scoping
right. Longer and more structured, but less reliable — the same "confidently
wrong, self-inconsistent" failure pattern flagged as the worst error in Step
8, now apparently *induced* by adding more (irrelevant) context rather than
just co-existing with good retrieval.

**Systematic citation-accuracy cost across the other 9 questions:** on every
question where the boosted-in document wasn't actually used (7 of the 9
single-hop AI-Act-article questions: Articles 5, 6(3), 10, 11, 12, 14, 113),
citation accuracy dropped — sometimes sharply (0.90 → 0.50) — because the
boosted document still gets listed as a citation even though the answer
never draws on it, the same mechanical "citations echo the retrieved set"
problem from Step 6's finding 1, now made worse simply because the boost
widens the retrieved set on more questions.

**Root cause, in plain terms:** the corpus-tag signal fires whenever *any*
retrieved chunk names another document, regardless of how central that
mention is to the actual question. An AI Act article that name-drops GDPR
in a boilerplate "without prejudice to..." clause triggers the same
full-strength boost as a chunk where the cross-reference is the entire point
— the mechanism has no notion of relevance, only presence.

**Not yet acted on — options for a future refinement:** restrict the
corpus-tag signal to only fire from a top-1 or top-2 ranked chunk (a passing
mention buried in a rank-5 chunk is far less likely to be question-relevant
than one in the top-ranked chunk); or require the boosted document to
actually get used in the final answer before crediting it in citations
(addressing the citation-accuracy cost directly, independent of the boost
question). No config change made yet — this step is measurement only.

### Step 14 — A real retrieval qrels file: true Precision/Recall, for free, forever

Every retrieval metric up to this point — Retrieval Hit, Chunk-Level Hit,
MRR — was judged against a single canonical "expected" location per
question, never against a complete list of every relevant chunk. That means
none of them were true Precision@k or Recall@k: Precision needs to know
what fraction of what was retrieved is actually relevant; Recall needs to
know what fraction of everything relevant was retrieved. Neither is
answerable without first knowing the full relevant set.

Built that ground truth. Exhaustively judging relevance across all 837
corpus chunks for all 41 questions (~34,000 judgments) isn't tractable, so
used the standard IR technique of **pooling**: for each question,
`eval/export_candidates_for_qrels.py` unions the top-20 results from 4
independent retrieval methods (dense-only, sparse-only, hybrid, and the live
pipeline with all current boosts) — no LLM calls, pure retrieval mechanics.
Mean pool size 33.4 chunks/question (1,369 total). Split into 7 batches
(`eval/split_qrels_candidates.py`) for careful judging, and had Claude
Sonnet 5 mark genuinely relevant chunks per a strict rubric
(`eval/QRELS_JUDGE_INSTRUCTIONS.md`) that explicitly warns against the exact
mistake Step 13 just found costing real accuracy — confusing topical
proximity with actual relevance.

`eval/merge_qrels.py` combined and validated the 7 judged batches: all 41
questions present exactly once, and — importantly — every judged-relevant
chunk cross-checked against its original candidate pool to catch any
hallucinated reference before it could corrupt downstream numbers. None
found. Result: `eval/qrels.jsonl`, mean 2.95 relevant chunks/question
(range 1-8, 121 total), zero questions with no relevant chunks in their pool.

**Caveat, stated for the record (also in the qrels instructions):** recall
computed against this file is bounded by the pool, not the full corpus — a
relevant chunk none of the 4 methods ever surfaced is invisible to it. This
is the accepted trade-off in real-world IR evaluation (TREC-style pooling),
not a shortcut unique to this project — just don't read a high recall number
as "we found everything in the corpus."

**What this unlocks — real numbers, and a free deterministic confirmation of
Step 13:** `eval/compute_retrieval_metrics.py` computes real Precision/Recall
for any retrieval config against the live pipeline, no LLM judging needed,
ever again, once the qrels file exists:

| Config | Precision | Recall | F1 |
|---|---|---|---|
| Contextual headers, no cross-reference boost | 0.356 | 0.676 | **0.427** |
| Contextual headers + cross-reference boost (current default) | 0.295 | 0.698 | **0.389** |

This is a clean, deterministic re-derivation of Step 13's expensive,
judged finding — the boost trades precision for a small recall gain, and the
net F1 effect is negative — obtained this time for free, from pure set
arithmetic, and now re-runnable for any future retrieval change without
spending another judging round. Recorded in Langfuse as
`phase3-hybrid-fixed500-ctxheaders-crossrefboost-precisionrecall-qrels` and
`phase3-hybrid-fixed500-ctxheaders-noboost-precisionrecall-qrels`.

**Also revealed, independent of the boost question:** even the *better*
(no-boost) configuration only reaches 0.356 precision — meaning roughly two
out of every three chunks the pipeline retrieves, on average, aren't
actually relevant to the question asked. This wasn't visible in any prior
metric (Retrieval Hit and Chunk-Level Hit only check "is at least one
relevant chunk present," never "how much of what came back was noise").
This is a new, general finding about the pipeline's precision — not
specific to the cross-reference boost — worth investigating on its own.

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
| + Contextual headers, judged by Claude Sonnet 5 on all 41 | independent judge, 6 metrics, full set | **36/41 doc-recall (87.8%, confirms the Groq-free screen); 34/41 (82.9%) fully/mostly correct — best full-set outcome-distribution result yet** |
| + GDPR cross-reference boost (keyword-triggered) | retrieval-hit only | 39/41 hit rate (95.1%), zero regressions |
| + Corpus-side cross-reference tagging (generalizes the above) | retrieval-hit only | **40/41 hit rate (97.6%) — new best** |
| Same boost, judged by Claude Sonnet 5 on the 15 questions it affects | independent judge, 6 metrics, boost-affected questions only | Retrieval Hit 73.3%→100%, Chunk-Level Hit 66.7%→93.3% (both up); **Answer Correctness 0.803→0.760, Citation Accuracy 0.617→0.510 (both down)** — mixed result, see Step 13 |
| First true Precision/Recall (qrels-based), no boost | deterministic, pooled relevance judgments | Precision 0.356, Recall 0.676, **F1 0.427** |
| Same, with cross-reference boost | deterministic, pooled relevance judgments | Precision 0.295, Recall 0.698, **F1 0.389 — confirms Step 13's finding for free, deterministically** |

## 5. What's still open

- **Investigate the pipeline's low precision, independent of the boost
  question** (Step 14, new) — even the better (no-boost) config only
  reaches 0.356 precision, meaning roughly 2 of every 3 retrieved chunks
  aren't actually relevant. No prior metric could see this. Worth its own
  investigation, separate from the boost's own precision problem.
- **Re-run `eval/compute_retrieval_metrics.py` after any future retrieval
  change** (Step 14) — it's now free (no LLM judging) and gives real
  Precision/Recall/F1 immediately, so there's no reason to skip it before
  deciding whether a change is worth keeping.
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
- **Fix the cross-reference boost's precision problem** (Step 13, new,
  actionable) — it fires on any retrieved chunk naming another document,
  regardless of whether that mention is actually central to the question.
  Caused a systematic citation-accuracy drop and 2 real answer-quality
  regressions (one a non-answer where a correct one existed before) across
  the 9 non-cross-reference questions it touches, alongside 4 genuine wins on
  actual cross-reference questions. Candidate fixes: only trigger from a
  top-1/top-2 ranked chunk's tag (not any of the top-5), or only credit a
  boosted document in citations if the answer actually uses it. **Now
  verifiable deterministically** — Step 14's qrels-based
  `eval/compute_retrieval_metrics.py` can check any candidate fix's real
  Precision/Recall/F1 immediately, without needing another judging round.
- **Even the generalized (Step 12) boost still requires a retrieved chunk to
  explicitly name the other document** — verified directly (see Step 12). A
  question whose best-matching chunks never happen to name the cross-referenced
  document in so many words remains unsolved; would need query
  reformulation/HyDE or a learned router instead.
- (Cheap, separate, low-impact, still open) exclude
  `gdpr_2016_679_mirror.html`'s 7 navigation-only chunks from ingestion —
  identified in Step 9, unrelated to the boost fix above.
- **Investigate the paragraph-38 chunking gap** (Step 10, new) — two
  Article 50 transparency-guidelines questions both miss the same specific
  passage, retrieved chunks stopping just short of it (paragraphs 34-37
  retrieved, 38 never surfaced) — looks like a systematic chunk-boundary
  issue for that document, not yet investigated.
- Contextual headers + recursive chunking stacked together — parked at the
  user's request until remaining tasks are complete.
- Phase 4 (ingestion sophistication — effective-date metadata, cross-reference
  linking) not started; the retrieval-hit misses (Digital-Omnibus-adjacent and
  cross-reference questions) remain the concrete targets, now with two
  independent judges' worth of evidence pointing at the same weak spots.
- Deployment (Render/Railway) deferred pending a separate GitHub account setup.
