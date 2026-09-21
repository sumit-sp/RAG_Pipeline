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

### Step 6 — External LLM-as-judge export (in progress)

Exported the live pipeline's output (hybrid retrieval + recursive chunking) against
all 41 questions — retrieved chunks with rank/score/text, generated answer,
citations, and the golden reference — to
`eval/pipeline_outputs_for_external_judge.jsonl`, for independent judging by Claude
Sonnet 5 rather than the Groq-family judge already in use (`gpt-oss-120b` judging
`gpt-oss-20b` shares a lab and model family, a self-grading-bias risk worth
cross-checking). **Results pending** — not yet folded back into this document.

## 4. Pass-rate timeline at a glance

| Stage | Checks used | Overall pass rate |
|---|---|---|
| Naive baseline (dense, first run, 2 crashes) | 4 pipeline | 26/41 (2 crashed) |
| Naive baseline (dense, bug-fixed) | 4 pipeline | 27/41 (66%) |
| + Hybrid search | 4 pipeline | 31/41 (76%) |
| + Reranking (reverted) | 4 pipeline | 26/41 (63%) |
| + Retrieval-hit & Answer-Correctness checks | 6 total | **24/41 (59%) — current reference point** |
| + Recursive chunking (Groq-free screen only) | retrieval-hit only | 34/41 hit rate (83%), full eval pending |

## 5. What's still open

- Fold the external (Claude) judge's results into this document once available,
  and decide whether recursive chunking becomes the default.
- Untried from Phase 3's list: contextual chunk headers.
- Phase 4 (ingestion sophistication — effective-date metadata, cross-reference
  linking) not started; the retrieval-hit misses (Digital-Omnibus-adjacent and
  cross-reference questions) are the concrete targets it should aim at.
- Deployment (Render/Railway) deferred pending a separate GitHub account setup.
