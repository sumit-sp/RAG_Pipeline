# Deployment Guide

> **Status:** this project has not yet been deployed publicly — this guide is
> for whoever does it (see `PROGRESS.md`: deployment was deferred pending a
> separate GitHub account). Everything here is written to actually be
> followed, not aspirational: it matches the current code, current env vars,
> and current recommended config as of `EVALUATION_HISTORY.md` Step 12.

## 1. What you're deploying

Two small services, deployed separately:
- **The API** (`app/api/main.py`, FastAPI) — the actual retrieval + generation
  pipeline, exposed as `/query` and `/health`.
- **The demo UI** (`demo/app.py`, Streamlit) — a thin frontend that just calls
  the API's `/query` endpoint.

Both read all their configuration from environment variables (no config
files, no hardcoded values) — see the table in §3.

## 2. Prerequisites

- **A GitHub repo** with this code pushed to it — Render and Railway both
  deploy via GitHub-connected auto-deploy, so this has to exist first.
- **A [Groq](https://console.groq.com) API key** (free tier is enough) —
  required, used for answer generation.
- **A Render or Railway account** (both have a free tier; steps for both are
  below).
- **Optional: a [Langfuse Cloud](https://cloud.langfuse.com) account** — for
  tracing/observability (see §7). Not required; the app runs fine without it.
- **Optional: a [Qdrant Cloud](https://cloud.qdrant.io) account** (or any
  reachable Qdrant server) — only needed if you want the vector index to
  survive redeploys instead of rebuilding every time (see §5).
- No other API keys. Embeddings run locally (no API key), and everything else
  the pipeline needs is already in the repo.

## 3. Environment variables reference

All of these are read in `app/core/config.py`. Only `GROQ_API_KEY` is
required — everything else has a working default.

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | *(required)* | Used for generation only — see `DECISIONS.md`, "Groq restricted to generation only" |
| `GENERATION_MODEL` | `openai/gpt-oss-20b` | The model used to write answers |
| `PIPELINE_BACKEND` | `plain` | Only `plain` is implemented so far |
| `EMBEDDING_BACKEND` | `local` | Only `local` is implemented so far — no API key needed |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Downloaded automatically on first use (from Hugging Face) |
| `QDRANT_URL` | *(unset)* | Set this to use a real/hosted Qdrant server (e.g. Qdrant Cloud) instead of the embedded/on-disk fallback — **see §5, this matters for deployment** |
| `QDRANT_API_KEY` | *(unset)* | Required if `QDRANT_URL` points at Qdrant Cloud (every request needs it there); leave unset for a local/docker-compose server with no auth configured |
| `QDRANT_PATH` | `./qdrant_local_data` | Only used when `QDRANT_URL` is unset |
| `QDRANT_COLLECTION` | `ai_act_corpus` | Base collection name (each retrieval mode gets its own suffix, see `collection_name()` in `config.py`) |
| `RETRIEVAL_MODE` | `hybrid` | `dense` or `hybrid` (dense + BM25 sparse) — **keep as `hybrid`**, it's the validated best (see `EVALUATION_HISTORY.md` Step 2) |
| `RETRIEVAL_TOP_K` | `5` | Chunks returned per query (before any cross-reference boost adds more — see below) |
| `HYBRID_PREFETCH_LIMIT` | `20` | Candidate pool size per signal (dense/sparse) before RRF fusion, hybrid mode only |
| `CHUNKING_STRATEGY` | `fixed` | `fixed` or `recursive` — keep as `fixed`, it's what the current best config (contextual headers) was built and validated against |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | `500` / `50` | Only relevant if you re-run ingestion with different values |
| `USE_CONTEXTUAL_HEADERS` | `false` | **Set this to `true` for deployment** — see the callout right below this table |
| `CONTEXTUAL_HEADERS_PATH` | `eval/contextual_headers.jsonl` | Already committed to the repo; no extra setup needed once `USE_CONTEXTUAL_HEADERS=true` |
| `USE_CROSS_REFERENCE_BOOST` | `true` | On by default — no action needed (see `EVALUATION_HISTORY.md` Steps 11-12) |
| `CROSS_REFERENCE_BOOST_LIMIT` | `5` | Extra chunks fetched per boosted document |
| `USE_RERANKING` | `false` | **Leave off** — tried and reverted, see `DECISIONS.md` |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | *(unset)* | Optional — tracing is a no-op if unset. See §7 |
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` | Only change this if self-hosting Langfuse instead of using Cloud |

**⚠️ Important gotcha:** `USE_CONTEXTUAL_HEADERS` defaults to `false`. The
current best-performing, validated pipeline configuration has it **on**
(87.8% → 97.6% retrieval-hit rate improvement across `EVALUATION_HISTORY.md`
Steps 7-12 depends on it). If you deploy with just the defaults, you'll get
a materially worse pipeline than what's documented as "current." Set
`USE_CONTEXTUAL_HEADERS=true` explicitly as an environment variable on
whatever platform you deploy to, and make sure it's also set when you run
ingestion (see §4) — it has to match at both ingestion time and query time,
since it's ingestion that actually bakes the headers into the stored chunks.

## 4. Ingestion — building the vector index

This has to run **once before the API can answer anything** (it populates
Qdrant), and again any time `data/raw/` or the chunking/header config changes.

```bash
USE_CONTEXTUAL_HEADERS=true python -m app.pipelines.plain.ingestion
```

This takes under a minute for the current 8-document, 837-chunk corpus. It
downloads the embedding model on first run (no API key, but does need
outbound internet access to Hugging Face the first time).

## 5. Vector store: ephemeral vs. persistent

**The default (embedded/on-disk Qdrant, `QDRANT_URL` unset) writes to local
disk.** Render and Railway's free/starter tiers use **ephemeral disks** —
that data is wiped on every redeploy or restart. Two options:

**Option A — rebuild on every deploy (simplest).** Run ingestion as part of
the build step (see §6's build command) so the index is always freshly built
from `data/raw/` for that deploy. Fine for this corpus's size — it's fast
and there's nothing to keep in sync. This is what the steps below use.

**Option B — persistent Qdrant (more production-realistic).** Stand up a
Qdrant Cloud free-tier cluster (free forever, no credit card: 0.5 vCPU /
1GB RAM / 4GB disk — comfortably enough for this project's ~5MB corpus) at
[cloud.qdrant.io](https://cloud.qdrant.io), or use any other reachable
Qdrant server with a real disk. Set `QDRANT_URL` to the cluster's URL and
`QDRANT_API_KEY` to its API key (both from the cluster's dashboard — Qdrant
Cloud requires the API key on every request, unlike a local/docker-compose
server), then run ingestion once by hand rather than on every deploy. The
index survives restarts and redeploys; you only re-run ingestion when the
corpus or chunking config actually changes.

## 6. Deploying the API (Render)

1. Push this repo to GitHub.
2. Render dashboard → **New → Web Service** → connect the GitHub repo.
3. **Build command:**
   ```
   pip install -r requirements.txt && USE_CONTEXTUAL_HEADERS=true python -m app.pipelines.plain.ingestion
   ```
   (drop the ingestion part if you went with Option B above)
4. **Start command:**
   ```
   uvicorn app.api.main:app --host 0.0.0.0 --port $PORT
   ```
5. Add environment variables in the Render dashboard: `GROQ_API_KEY`,
   `USE_CONTEXTUAL_HEADERS=true`, and `QDRANT_URL` + `QDRANT_API_KEY` if using
   Option B. Everything else can be left at its default (see the table in §3).
6. Deploy. Once it's live, note the public URL Render assigns — the demo
   needs it.
7. Sanity check:
   ```bash
   curl -s https://<your-render-url>/health
   curl -s -X POST https://<your-render-url>/query \
     -H "Content-Type: application/json" \
     -d '{"question": "When do Annex III high-risk AI system obligations become applicable?"}'
   ```

## 6b. Deploying the demo (Render, second service)

1. Same repo, new Render Web Service.
2. **Build command:** `pip install -r requirements.txt`
3. **Start command:**
   ```
   streamlit run demo/app.py --server.port $PORT --server.address 0.0.0.0
   ```
4. Environment variable: `API_URL=<the API service's public URL from step 6>`

## 6c. Railway (alternative to Render)

Same shape, different dashboard: **New Project → Deploy from GitHub repo** —
Railway auto-detects Python. Set the same build/start commands and
environment variables per service (§6 and §6b) in the Railway dashboard.
Deploy the API and the demo as two separate services within the same Railway
project, wired together with `API_URL` the same way as above.

## 7. Langfuse Cloud tracing (optional)

Tracing is entirely optional and a no-op if unset — the app works fine
without it. If you want it (see `EVALUATION_HISTORY.md` for what it's used
for — tracking eval history and generation traces):

1. Create a free account and project at [cloud.langfuse.com](https://cloud.langfuse.com).
2. Get your project's public/secret API keys from Settings → API Keys.
3. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` as environment
   variables on the API service (not the demo — only the API makes
   generation calls).

**Note:** `docker-compose.yml` in this repo defines a full **self-hosted**
Langfuse stack (Postgres, ClickHouse, Redis, MinIO). That was the original
Phase 0 scaffold, but the project switched to **Langfuse Cloud** instead
(see `DECISIONS.md`, "Langfuse Cloud adopted now, not deferred to Phase 5")
— you do not need to run any of those containers. `docker-compose.yml` is
kept for local Qdrant-via-Docker only, if you ever want that instead of the
embedded mode; ignore its Langfuse services.

## 8. GitHub Actions CI

Once pushed to GitHub, add `GROQ_API_KEY` as a repository secret (**Settings
→ Secrets and variables → Actions**) so `.github/workflows/eval.yml` can run
on every push. This workflow only runs the Groq-free, deterministic
retrieval-hit check (`eval/test_retrieval_hit.py`) — no LLM-judged eval runs
automatically (see `DECISIONS.md`, "CI eval scope narrowed"), so
`GROQ_API_KEY` isn't actually required for CI to pass, only for the deployed
app itself to answer questions.

> **Known gap, not yet fixed:** `.github/workflows/eval.yml`'s ingestion step
> does not currently set `USE_CONTEXTUAL_HEADERS=true`. That means CI is
> presently testing the pre-Step-7 pipeline configuration, not the current
> best one — its retrieval-hit numbers won't match what's documented in
> `EVALUATION_HISTORY.md`, and it may fail against `test_retrieval_hit.py`'s
> current `MIN_HIT_RATE = 0.93` threshold (calibrated for the *with-headers*
> pipeline). Worth fixing before relying on this workflow as a real
> regression gate.

## 9. Post-deploy checklist

- [ ] `GET /health` on the API returns `{"status": "ok", ...}`
- [ ] `POST /query` with a real question returns a grounded answer with
      `sources`
- [ ] The demo UI loads and successfully calls the API (check `API_URL` is
      set correctly)
- [ ] If using Option A (rebuild-on-deploy): confirm the build logs show
      "Indexed 837 chunks..." completing without errors
- [ ] If using Langfuse: a query shows up as a trace in the Langfuse Cloud
      dashboard within a few seconds

## 10. Troubleshooting

**Ingestion appears to hang indefinitely with no error, at or right after
model loading (~0% CPU, no progress for minutes).** This is a known conflict
between `torch` (used by `sentence-transformers` for dense embeddings) and
`onnxruntime` (used by `fastembed` for BM25 sparse embeddings, hybrid mode
only) — both bundle their own Intel OpenMP runtime, and loading both into
the same process can deadlock silently on some machines instead of erroring.
Observed directly on Windows: encoding a real batch of chunks blocked with
essentially zero CPU time for many minutes, no exception, no timeout.
**Already fixed automatically** — `app/core/config.py` sets
`KMP_DUPLICATE_LIB_OK=TRUE` (the standard workaround) at import time, before
any embedding code loads, so this shouldn't recur. If it somehow still does
on some environment, also try setting `OMP_NUM_THREADS=1` — that wasn't
actually necessary when this was diagnosed (confirmed via isolated testing:
`KMP_DUPLICATE_LIB_OK=TRUE` alone was sufficient), but limits thread-pool
contention further as a fallback, at some cost to encoding speed.

**Qdrant Cloud returns `400 Bad Request: Index required but not found for
"source_doc"` on any query.** The embedded/on-disk local Qdrant mode
silently allows filtering on a payload field with no index; Qdrant Cloud (and
likely any real Qdrant server) requires an explicit payload index before a
`Filter`/`FieldCondition` can use that field — this only surfaces once the
cross-reference boost (`app/pipelines/plain/retrieval.py`) tries to filter
on `source_doc`. **Already fixed** — `ensure_collection`/
`ensure_hybrid_collection` in `app/pipelines/plain/vector_store.py` now
create the needed index automatically whenever a *new* collection is
created. If you're hitting this on a collection that already existed before
this fix, create the index once by hand:
```python
from qdrant_client.models import PayloadSchemaType
from app.pipelines.plain.vector_store import get_qdrant_client
get_qdrant_client().create_payload_index(
    collection_name="<your collection name>",
    field_name="source_doc",
    field_schema=PayloadSchemaType.KEYWORD,
)
```
