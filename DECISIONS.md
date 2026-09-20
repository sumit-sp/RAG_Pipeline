# Decisions log

## Phase 3

### Bug fix: empty Groq responses from gpt-oss reasoning-token exhaustion
Both the Phase 2 baseline and an early Phase 3 hybrid-search run showed 2/41 questions returning an empty `actual_output` and erroring out of DeepEval scoring entirely. Root-caused by direct inspection of the raw Groq response: `gpt-oss-20b` (and `gpt-oss-120b`, used as judge) are reasoning models that spend part of their token budget on hidden "reasoning tokens" before the visible answer; on some prompts this exhausted the default 2048-token completion budget entirely (`finish_reason="length"`, `reasoning_tokens=2046` of `2048`), leaving zero tokens for the actual answer.
**Fix:** `max_completion_tokens=4096` and `reasoning_effort="low"` on every Groq call (`app/pipelines/plain/generation.py`, `eval/judge_model.py`), plus `PlainGenerator.generate()` now raises a visible `RuntimeError` instead of silently returning empty text if this ever recurs.
**Why this matters for Phase 3 specifically:** this bug was corrupting the retrieval-mode comparison (different runs lost different questions to the crash, which isn't a fair like-for-like comparison), so it had to be fixed and both dense and hybrid re-measured before trusting any before/after numbers.

### Discovered: real run-to-run scoring variance on identical config
Re-running the dense-mode eval twice (once before, once after the bug fix above — same retrieval mode, same code otherwise) produced meaningfully different metric means: faithfulness 0.952 → 0.887, contextual precision 0.759 → 0.691. Both runs used `temperature=0.0` throughout. This is a known characteristic of served, batched LLM inference (Groq's MoE serving isn't bit-exact at temperature 0), and it applies to both the generator and the DeepEval judge model, since the judge makes its own LLM calls internally to score each case.
**How to apply:** individual metric means can swing ~0.05-0.07 between identical-config runs — that's the noise floor. Phase 3 (and later Phase 4) comparisons should trust large, consistent swings (especially the overall pass/fail count, which needs a bigger shift to move) over small deltas in any single metric's third decimal place. Where a result is close to this noise floor, say so rather than claiming a clean win.

### Kept: hybrid search (dense + BM25 sparse, RRF fusion) over dense-only
Implemented via Qdrant's native named-vector + sparse-vector support (`app/pipelines/plain/vector_store.py`, `sparse_embedding.py` using fastembed's local `Qdrant/bm25` model — no new API key). Compared against the bug-fixed dense baseline on the same 41 questions:
- Overall pass rate (all 4 metrics ≥ 0.5): **27/41 (66%) → 31/41 (76%)**, a swing well above the observed noise floor.
- Contextual precision on multi-hop questions: 0.667 → 0.844. On cross-reference questions: 0.770 → 0.826.
- Faithfulness: 0.887 → 0.931. Answer relevancy dropped slightly: 0.989 → 0.917 (still comfortably above threshold).
**Why:** dense-only embeddings tend to match on semantic topic similarity, which can miss exact-term matches (article numbers, defined terms) that BM25 catches directly — exactly the failure mode multi-hop/cross-reference questions hit, since they need chunks pulled in by different signals from two different documents. Kept as the new default (`RETRIEVAL_MODE=hybrid`).
**Trade-off accepted:** adds `fastembed` as a dependency and roughly doubles per-chunk ingestion cost (two embeddings per chunk instead of one) — negligible for an 8-document corpus, would need reconsidering at much larger scale.

### Tried and reverted: cross-encoder reranking on top of hybrid search
Implemented as an orthogonal post-processing step (`app/pipelines/plain/reranking.py`, `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers, local, no API key): fetch a wider candidate set (20) from whatever retrieval mode is active, rerank with the cross-encoder, keep the top-k. Tested on top of hybrid search (the current best) on the same 41 questions:
- Overall pass rate: **31/41 (76%) → 26/41 (63%)** — a real regression, well above the ~0.05-0.07 noise floor.
- Mixed and inconsistent per-metric/per-difficulty effects: answer relevancy improved (0.917→0.984) and contextual recall improved (0.776→0.809), but contextual precision dropped (0.754→0.718), and multi-hop specifically got much worse (precision 0.844→0.602) while cross-reference and temporal-conflict improved. Net effect: a different, smaller set of questions pass all four metrics simultaneously.
**Reverted** — `USE_RERANKING` defaults to `false`. The code is kept (not deleted) since the plumbing is reusable if a different reranker is worth trying later, but it's off by default.
**Plausible explanation, not confirmed:** `ms-marco-MiniLM-L-6-v2` is trained on general web/MS-MARCO passage ranking, not legal/regulatory text. The hybrid retrieval it's reranking already combines dense semantic matching with BM25's exact-term matching (article numbers, defined terms) — precisely the signal this domain needs — and the generic cross-encoder's reordering appears to work against that rather than refine it, especially for multi-hop questions needing two specific documents together.
**Why keep this in the write-up:** this is exactly the kind of negative result the project spec asks to document — "tried X, it made things worse because Y, reverted" is worth as much as a positive result for demonstrating real evaluation discipline rather than just reporting whichever numbers look best.

### README kept live throughout, not deferred to Phase 5
The spec's own plan defers the full README (architecture diagram, results table, how-to-run, known limitations) to Phase 5. The user explicitly asked for it to be updated continuously instead, wanting a complete, runnable local setup plus deployment guidance documented before deployment happens, not written retroactively at the end.
**How to apply:** README gets updated at the end of each phase from here on with current architecture, setup steps, config, and results — Phase 5 becomes "finalize" (polish + failure-modes writeup) rather than "write from scratch."

## Phase 2

### DeepEval judge model: Groq gpt-oss-120b, not OpenAI's default
DeepEval's built-in metrics (faithfulness, answer relevancy, contextual precision/recall) default to an OpenAI model as judge, which needs a key we don't have. Rather than default to that or reuse the same model that generates answers (gpt-oss-20b — self-grading bias, a known LLM-as-judge pitfall), used the larger `gpt-oss-120b` via the existing Groq key as an independent judge, via a small custom `DeepEvalBaseLLM` wrapper (`eval/judge_model.py`).
**Trade-off accepted:** both generator and judge are still GPT-OSS-family models from the same lab, so some correlated bias risk remains — a truly independent frontier judge (e.g. Claude or GPT-4o) would be stronger, but wasn't available without procuring another API key. Noted here so eval numbers are read with that caveat, not treated as bulletproof.

A running record of non-trivial decisions, including rejected alternatives and negative results, logged as they're made (not retroactively).

## Phase 0

### Domain: EU AI Act compliance assistant
Kept the spec's default domain rather than swapping to the multilingual GDPR or SEC-filings alternatives. Reasoning per the spec: current and specific to European employers (GPAI transparency enforcement began 2 August 2026; Digital Omnibus restructured the high-risk timeline as of 27 July 2026), spans multiple cross-referencing documents, and includes a genuine, checkable versioning test case (Omnibus-amended dates vs. the original Regulation text).

### Embeddings: local model instead of a hosted API, deferred to Phase 1
The spec's tech-stack table suggests picking one hosted embedding API (OpenAI / Voyage / Cohere) up front. No API keys or accounts are set up yet, and procuring one isn't a Phase 0 task — rather than block on that, Phase 1 will use a local, no-API-key embedding model (sentence-transformers, likely `BAAI/bge-small-en-v1.5`) behind a small `Embedder` abstraction, so switching to a hosted provider later is a config/env-var change, not a rewrite of ingestion or retrieval code.
**Trade-off accepted:** local embedding quality is generally a notch below the best hosted options, and CPU inference is slower — acceptable for a corpus this size (8 documents) and consistent with the spec's own note that embedding choice "is not the differentiator — don't over-invest here early."
**Revisit:** if Phase 3 eval numbers show retrieval quality bottlenecked specifically by embedding quality (not chunking/reranking), swap in a hosted provider and re-run the eval set for an apples-to-apples comparison.

### EUR-Lex documents sourced from Wayback Machine snapshots, not a live fetch
Live `eur-lex.europa.eu` sits behind an AWS WAF bot-challenge that blocked every direct fetch attempt (HTML and PDF format alike), and the org's browser policy separately blocks that domain outright. Rather than give up on the citation-of-record source or substitute an unofficial mirror for the full Regulation text, pulled each of the 3 EUR-Lex documents (AI Act, Digital Omnibus, GDPR) from the Internet Archive's Wayback Machine, which serves its own previously-crawled cache unaffected by the live-site challenge. Verified each snapshot's `<title>` and article count against expectations before accepting it.
**Trade-off accepted:** these are point-in-time snapshots rather than a live pull — flagged in `data/raw/SOURCES.md` and `PROGRESS.md` to re-diff against a live fetch later if network access to EUR-Lex ever becomes available, in case of consolidation/corrigendum changes since the snapshot date. The EUR-Lex URL (not the Wayback mirror URL) still gets used as the citation-of-record link surfaced in the UI later, per the spec's guidance.

### No accounts/credentials procured yet
Docker, an embedding-provider API key, and a Render/Railway account are all unset up. None are required for Phase 0 (pure scaffolding + document collection), so this isn't a blocker — flagged here so Phase 1 planning accounts for needing at least Docker (for Qdrant + Langfuse) before ingestion can run end-to-end.
