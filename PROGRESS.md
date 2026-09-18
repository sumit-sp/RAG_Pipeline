# Progress log

## Phase 0 — Scope & scaffold (in progress)

- [x] Domain confirmed: EU AI Act compliance assistant (spec default)
- [x] Repo structure scaffolded per spec
- [x] Source documents downloaded into `data/raw/` — all 8 spec items in place (25 files total: EUR-Lex + mirror copies for the AI Act/GDPR, 4 guidance PDFs, a curated 15-Article Service Desk subset). See `data/raw/SOURCES.md` for the full manifest and one caveat: the 3 EUR-Lex documents came from Wayback Machine snapshots (live EUR-Lex is behind a bot-challenge WAF and also blocked by org browsing policy) — worth a diff against a live fetch later if that network access ever opens up. One guidance PDF (AI-generated-content Code of Practice) is a "second draft", not a confirmed final version — flagged for a recheck before Phase 5.
- [x] `docker-compose.yml` drafted (Qdrant + Langfuse self-host stack) — not yet run; Docker not confirmed installed locally
- [x] README stub written
- [x] `DECISIONS.md` / `PROGRESS.md` initialized

Phase 0 exit criteria met: repo scaffolded, all 8 documents in place under `data/raw/`, README stub committed.

Next: start Phase 1 (walking skeleton) only after explicit go-ahead.
