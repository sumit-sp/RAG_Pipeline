# Progress log

## Phase 0 — Scope & scaffold (in progress)

- [x] Domain confirmed: EU AI Act compliance assistant (spec default)
- [x] Repo structure scaffolded per spec
- [x] Source documents downloaded into `data/raw/` — all 8 spec items in place (25 files total: EUR-Lex + mirror copies for the AI Act/GDPR, 4 guidance PDFs, a curated 15-Article Service Desk subset). See `data/raw/SOURCES.md` for the full manifest and one caveat: the 3 EUR-Lex documents came from Wayback Machine snapshots (live EUR-Lex is behind a bot-challenge WAF and also blocked by org browsing policy) — worth a diff against a live fetch later if that network access ever opens up. One guidance PDF (AI-generated-content Code of Practice) is a "second draft", not a confirmed final version — flagged for a recheck before Phase 5.
- [x] `docker-compose.yml` drafted (Qdrant + Langfuse self-host stack) — not yet run; Docker not confirmed installed locally
- [x] README stub written
- [x] `DECISIONS.md` / `PROGRESS.md` initialized

Phase 0 exit criteria met: repo scaffolded, all 8 documents in place under `data/raw/`, README stub committed.

## Phase 1 — Walking skeleton (in progress)

- [x] `app/core/interfaces.py` — Ingestor/Retriever/Generator Protocols (plus a small Embedder protocol, since embeddings needed to be swappable per an explicit ask)
- [x] Naive parsing — BeautifulSoup for `.html`, PyMuPDF for `.pdf`, no per-document special-casing
- [x] Fixed-size chunking — 500 tokens / 50 overlap (tiktoken `cl100k_base`), own module for Phase-5 testability
- [x] Embed + store in Qdrant — local sentence-transformers (`BAAI/bge-small-en-v1.5`), **Qdrant in embedded/on-disk mode (`qdrant_local_data/`) — no Docker needed.** Swapping to the docker-compose Qdrant server later is just setting `QDRANT_URL`.
- [x] Retrieval — top-k dense search only, via `qdrant_client.query_points`
- [x] Generation — Groq-hosted `openai/gpt-oss-20b`, context-grounded, cites `source_doc` per chunk used
- [x] FastAPI `/query` and `/health` endpoints
- [x] Minimal Streamlit UI, tested live in-browser against the running API
- [x] Ingestion run: **837 chunks indexed** from the Phase 0 corpus
- [x] Manually verified against the spec's own temporal-conflict example question ("When do Annex III high-risk AI system obligations become applicable?") — correctly answered **2 December 2027** (the Omnibus-amended date), citing the AI Act, the Digital Omnibus, and Article 6 guidance together. This is a qualitative spot-check, not an eval number — Phase 2 builds the real harness.
- [ ] Deploy to Render or Railway — **explicitly deferred** by the user. They're setting up a separate GitHub account to push this repo to before connecting it to Render/Railway; that account setup and the push/deploy itself will happen in a later session. Code is deploy-ready in the meantime (all config via env vars, `requirements.txt` present, no hardcoded paths).

Known rough edges to leave for Phase 2/3/4 rather than fix now: no reranking/hybrid search yet (dense-only, as specced), no confidence checks on retrieval, and generation runs on Groq's fast open-weight `gpt-oss-20b` rather than a larger frontier model — worth comparing against `gpt-oss-120b` or a hosted frontier model in Phase 3 if faithfulness scores come back borderline.

Next: get explicit go-ahead before Phase 2 (eval harness) — Phase 1 isn't fully closed out until deployment happens or is explicitly deferred.

## Phase 2 — Eval harness (in progress)

- [x] Golden set drafted (43 candidates: 6 hand-verified by Claude directly against primary source text — the Digital Omnibus amendment vs. original AI Act Article 113 — plus 37 agent-drafted from the rest of the corpus) and reviewed by the user. **User rejected 2 (candidates #29 and #32 in the review doc — GPAI scope guidelines designation-timing inference, and the draft Code of Practice status question)**, approved the remaining 41. Final set: `eval/golden_set.jsonl`, 41 questions.
- [x] `eval/judge_model.py` — custom DeepEval model wrapping Groq `gpt-oss-120b` as judge (see DECISIONS.md for the self-grading-bias trade-off)
- [x] `eval/run_eval.py` — pytest + `assert_test()` against faithfulness/answer relevancy/contextual precision/contextual recall, threshold 0.5, run live against the plain-backend pipeline per question
- [x] `.github/workflows/eval.yml` — runs `deepeval test run` on push (untested until there's a GitHub remote; local runs use plain `pytest` instead — see below)
- [x] Baseline numbers in `eval/results.md` — **26/41 (63%) pass all four metrics simultaneously.** Faithfulness 0.95 and answer relevancy 0.98 are strong; contextual precision 0.76 and recall 0.75 are the bottleneck, worst specifically on cross-reference questions (0.63/0.54) — dense-only retrieval struggles to pull chunks from two different documents at once. This is the concrete target for Phase 3. Raw per-question scores in `eval/eval_run_raw_results.jsonl`.
- [x] Found and logged: 2/41 questions (both about the GPAI Safety & Security Code of Practice chapter) got an empty Groq generation response and errored out of scoring entirely rather than just scoring low — not yet root-caused, noted in `eval/results.md` as a Phase 3+ robustness item (handle empty/`None` LLM responses explicitly).
- Known local-environment quirk: `deepeval test run` hangs indefinitely on this network (looks like a blocked telemetry/version-check call). Plain `pytest eval/run_eval.py` runs the identical suite without hanging — used for all local runs. CI keeps `deepeval test run` since GitHub Actions runs on a different network.

Phase 2 exit criteria met: baseline eval numbers exist, committed, and CI is wired to run them automatically on push (untestable until a GitHub remote exists — tracked as part of the deferred deployment work).

## Phase 3 — Retrieval & generation quality (in progress)

- [x] Found and fixed a real bug while setting up before/after comparisons: `gpt-oss-20b`/`gpt-oss-120b` sometimes exhausted their token budget on hidden reasoning tokens before emitting an answer (`finish_reason="length"`, empty content). Fixed with `max_completion_tokens=4096` + `reasoning_effort="low"` on every Groq call, and `PlainGenerator` now raises instead of silently returning empty text. Details in `DECISIONS.md`.
- [x] Discovered and documented real run-to-run scoring variance (~0.05-0.07 per metric) between identical-config re-runs — a genuine characteristic of served LLM inference at `temperature=0`, not a bug. Documented as a noise floor for reading all Phase 3/4 comparisons in `eval/results.md`.
- [x] **Hybrid search (dense + BM25 sparse via Qdrant native support) — KEPT.** Pass rate 27/41 → 31/41 (66%→76%) against the bug-fixed dense baseline. `RETRIEVAL_MODE=hybrid` is now the default. Full numbers in `eval/results.md`.
- [x] **Reranking (cross-encoder on top of hybrid) — TRIED AND REVERTED.** Overall pass rate went 31/41 → 26/41 despite 2 of 4 metrics improving on average — the metrics that got worse (precision, multi-hop specifically) knocked out different questions than before. `USE_RERANKING` defaults to `false`; code kept, not deleted. Full write-up in `eval/results.md` and `DECISIONS.md`. This is the phase's documented negative result.
- [ ] Contextual chunk headers — not yet tried
- [ ] Chunk size / overlap tuning — not yet tried

Phase 3 exit criteria met: eval numbers meaningfully improved over the Phase 2 baseline (27/41 → 31/41 via hybrid search), a results table exists, and a negative result is documented (reranking).

- [x] **Eval harness extended** with a component-level retrieval-hit check (reference-based, programmatic) and an application-level Answer Correctness check (GEval vs. golden answer) — prompted by checking the suite against a standard component/pipeline/application RAG-eval framework and finding it only covered the pipeline level. Retrieval-hit rate: 80% (33/41) — lower than the LLM-judged contextual metrics suggested, since those can score well even when a different-than-expected chunk covers the same ground. Combined pass rate across all 6 checks: **24/41 (59%)**, now the reference point for future comparisons (not the earlier 31/41, which used a narrower 4-check definition). Full write-up in `eval/results.md` and `DECISIONS.md`.

- [x] Chunk-size/strategy tuning (Groq-free, retrieval-hit only): fixed 500/50 (baseline) 80%, fixed 800/100 78% (flat/worse), **recursive 500/50 83%** (best, real/deterministic since this check has no LLM involved).
- [x] Exported full pipeline outputs (hybrid retrieval + recursive chunking, all 41 questions) to `eval/pipeline_outputs_for_external_judge.jsonl`, judged independently by Claude Sonnet 5 (`eval/rag_eval_report.xlsx`). Results folded into `EVALUATION_HISTORY.md` (Step 6) and `DECISIONS.md`: 34/41 (83%) doc-level retrieval recall — exact match with the Groq-free screen, cross-confirming that result. Surfaced new, concrete findings: citations just echo retrieved docs (no independent signal), document-level recall hides chunk-level misses, one clear hallucination under good retrieval, a self-consistency failure across near-duplicate questions, and boilerplate nav text diluting Service Desk chunks.
- [x] Created `EVALUATION_HISTORY.md` — the single consolidated narrative (corpus/goal, naive baseline, every improvement tried with before/after numbers) that was previously scattered across README/PROGRESS/DECISIONS/results.md.

## Standing rules (2026-09-21)

- **Groq restricted to generation only** — no more Groq for judging or any other LLM-assist task.
- **LLM-judge / LLM-content-generation tasks now go through a JSON export**, run externally through Claude Sonnet 5 by the user, results brought back manually. Established pattern: `eval/export_for_external_judge.py` (already used once, Phase 3 Step 6).
- Following from this: `eval/judge_model.py` and the Groq-judge metrics in `eval/run_eval.py` are kept as an optional manual tool, **not run in CI or by default**. `.github/workflows/eval.yml` now runs only `eval/test_retrieval_hit.py` — Groq-free, deterministic, no API key needed, gated on an aggregate hit-rate floor (75%) rather than per-question so it doesn't stay red over already-documented gaps.

## Phase 3 — Contextual chunk headers (in progress, awaiting external processing)

- [x] `eval/export_chunks_for_contextual_headers.py` — exports every source document's full text + its own chunk list (25 documents, 837 chunks, current default fixed-500/50 chunking) to `eval/chunks_for_contextual_headers.jsonl`, with task instructions in `eval/CONTEXTUAL_HEADERS_INSTRUCTIONS.md`, for the user to run through Claude Sonnet 5 separately.
- [x] Ingestion-side consumer ready: `app/pipelines/plain/contextual_headers.py` + `USE_CONTEXTUAL_HEADERS`/`CONTEXTUAL_HEADERS_PATH` config, wired into `PlainIngestor` — prepends the returned header to each chunk before embedding. Tested with a fake header; verified correct chunk targeting. Does nothing until `eval/contextual_headers.jsonl` (837 lines expected) comes back.
- [ ] Awaiting the user's Sonnet-5 output, then: re-ingest with headers on, re-run the Groq-free retrieval-hit check, and (per the new rule) get judged results via another external export rather than the Groq judge.

Next: get explicit go-ahead before continuing. Other open items: run recursive chunking through a same-config comparison; act on the Step-6 findings (citation attribution, chunk-level recall metric, strip Service Desk boilerplate); or move to Phase 4.
