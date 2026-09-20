# Decisions log

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
