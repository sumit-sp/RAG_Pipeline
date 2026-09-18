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
