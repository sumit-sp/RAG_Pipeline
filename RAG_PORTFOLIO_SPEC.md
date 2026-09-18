# RAG Portfolio Project — Build Spec

## Instructions for the coding agent (read this first)

This spec is phased on purpose. The person you're building this with has a documented failure pattern: going deep on document parsing/ingestion and never reaching evaluation or deployment. This spec exists to structurally prevent that. Follow these rules regardless of what else you're asked mid-project:

1. **Work through phases in order: 0 → 1 → 2 → 3 → 4 → 5 → (6, optional).** Do not start a phase's tasks until the previous phase's "Exit criteria" are met. Phase 6 additionally requires Phase 5's Definition of Done to be fully checked off first — never start it as a way to avoid finishing Phase 5.
2. **If asked to improve ingestion/extraction before Phase 3 is deployed and has eval numbers attached, push back.** Point to this spec, name the current phase, and ask whether they want to explicitly override the sequence (they might have a good reason — but make them say so, don't default to it).
3. **No quality claim without a number.** "This should retrieve better" is not acceptable after Phase 2 exists. Every change in Phase 3/4 gets a before/after eval score in `eval/results.md`.
4. **Every phase ends with something runnable**, not just code that should work. If a phase is running long, ship the smaller version and note the gap in `PROGRESS.md` rather than polishing in place.
5. Log non-trivial decisions in `DECISIONS.md` as you make them, including rejected alternatives and negative results — not retroactively at the end.
6. **Optimize for readability and simplicity over cleverness or unnecessary abstraction, in the `plain` backend especially.** The person needs to be able to understand and personally debug every important piece of this code. If a design pattern, indirection layer, or framework feature doesn't earn its complexity for this specific system, don't add it — when in doubt, write the more boring, more obvious version.

---

## Project overview

**Goal:** a portfolio-quality RAG system demonstrating senior/lead-level AI engineering judgment — not just a working chatbot, but evidence of evaluation discipline, production awareness, and documented trade-off decisions.

**Domain (default — edit this before starting if you want a different one):**
An EU AI Act compliance assistant — a RAG system over the AI Act and its supporting regulatory documents (full list in Phase 0) that answers questions like "which obligations apply to my AI system, and when."
*Why:* current and specific to European employers — GPAI transparency obligations became enforceable with real fines on 2 August 2026, and the high-risk timeline was just restructured by the "Digital Omnibus" (in force since 27 July 2026). The corpus spans multiple distinct, cross-referencing documents rather than one regulation that fits in a single context window, and includes a genuine versioning challenge: the Omnibus amendment changed dates written into the original Regulation, which is a real, checkable test of whether the system surfaces the current obligation or a stale one.
*Swap-in alternatives:* (a) a genuinely multilingual EU corpus (GDPR guidance across a couple of national DPAs, in German/French/English); (b) SEC 10-K/10-Q filings — a legitimate general choice, but a distinctly American artifact if targeting European employers specifically.

**Non-goals for v1:** see Appendix.

---

## Definition of done (whole project)

- [ ] Live deployed URL (not just a repo you have to clone to see working)
- [ ] Public GitHub repo, clean structure (see below)
- [ ] README with: architecture diagram, results table (baseline vs. each improvement, with eval numbers), an honest "known limitations / failure modes" section
- [ ] Tests + GitHub Actions CI running the eval suite on every push
- [ ] Langfuse tracing wired into every LLM call
- [ ] A basic, demonstrated prompt-injection defense
- [ ] `DECISIONS.md` readable top-to-bottom as a narrative of real trade-offs
- [ ] Short write-up (blog post or LinkedIn post) on 2–3 decisions and their trade-offs — draft this after Phase 5
- [ ] (Optional/stretch, see Phase 6) A documented plain-vs-LangChain comparison on the same eval set

---

## Repo structure (set this up in Phase 0, don't reorganize later)

```
rag-portfolio/
├── app/
│   ├── core/
│   │   ├── interfaces.py   # Ingestor / Retriever / Generator contracts — define once, shared by both backends
│   │   └── models.py       # shared dataclasses (Chunk, RetrievedContext, Answer, ...)
│   ├── pipelines/
│   │   ├── plain/          # hand-rolled backend — the only one built in Phases 1–5
│   │   │   ├── ingestion.py
│   │   │   ├── retrieval.py
│   │   │   └── generation.py
│   │   └── langchain/      # LangChain backend — Phase 6, stretch, after Phase 5 is done
│   │       ├── ingestion.py
│   │       ├── retrieval.py
│   │       └── generation.py
│   └── api/                # FastAPI app; PIPELINE_BACKEND env var selects plain vs langchain
├── eval/
│   ├── golden_set.jsonl
│   ├── run_eval.py
│   ├── results.md
│   └── backend_comparison.md   # Phase 6 only
├── demo/                 # Streamlit or Gradio app
├── tests/
├── data/
│   └── raw/              # source documents (gitignored if large)
├── .github/workflows/
│   └── eval.yml
├── docker-compose.yml    # Qdrant + Langfuse for local dev
├── DECISIONS.md
├── PROGRESS.md
└── README.md
```

---

## Tech stack (don't substitute without logging why in DECISIONS.md)

| Layer | Tool | Notes |
|---|---|---|
| Parsing | HTML parsing (BeautifulSoup/markdown-it, preserving heading hierarchy) for the EUR-Lex/artificialintelligenceact.eu/gdpr-info.eu sources; PyMuPDF for the PDF guidance documents (items 3–6 in Phase 0) | Most of this corpus is already-structured HTML — exploit that instead of treating it like a scanned PDF |
| Vector store | Qdrant (local Docker) | Native hybrid search, correct metadata pre-filtering |
| Embeddings | Pick one API (OpenAI / Voyage / Cohere) and record why | Not the differentiator — don't over-invest here early |
| Evaluation | DeepEval + custom golden set | Pytest-native (`assert_test()`, metric thresholds) — plugs directly into the CI gate below, more naturally than Ragas's dataset-oriented workflow |
| API | FastAPI | |
| Demo UI | Streamlit (Gradio is an equally valid swap) | Chainlit looks like the better fit on paper (built-in source tracing, feedback, streaming) but its founding team stepped back from active development in May 2025 and two high-severity CVEs surfaced in late 2025 — avoid it for now |
| Observability | Langfuse (self-hosted via Docker) | Free, MIT-licensed, tracing + cost + eval in one |
| Deployment | Render or Railway | GitHub-connected auto-deploy, free/cheap tier — not AWS/GCP/Azure; raw cloud infra setup is a scope-creep trap, not a skill this project needs to demonstrate |
| CI | GitHub Actions | Runs `deepeval test run` on push |

**Deployment philosophy:** containerize everything (Docker), keep all config in env vars, and avoid provider-specific SDKs for core logic — this gets real portability across Render/Railway/any cloud's container service without building a dedicated abstraction layer for clouds you're not actually deploying to. Don't chase "fully cloud-agnostic" as its own goal; that's speculative complexity for a problem this project doesn't have.

---

## Pipeline architecture: two interchangeable backends

Ingestion, retrieval, and generation are each built twice, behind one shared interface, so they're comparable and swappable rather than tangled together:

- `app/core/interfaces.py` — defines the contract (e.g. `Ingestor`, `Retriever`, `Generator` as Python Protocols) that both backends satisfy. Define this once, early, and don't change it just to make one backend's life easier.
- `app/pipelines/plain/` — hand-rolled implementation using the libraries already in the tech stack directly, no orchestration framework. **This is the only backend built during Phases 1–5** — it's what you'll actually iterate on, debug, and be able to explain line-by-line in an interview.
- `app/pipelines/langchain/` — a LangChain-based implementation of the same three stages, satisfying the same interface. Built in **Phase 6, only after Phase 5's Definition of Done is fully checked off** — not in parallel with the main build.
- A `PIPELINE_BACKEND` env var (`plain` | `langchain`) selects which implementation the API layer uses at runtime.

**Why sequenced this way, not built together from the start:** building both from Phase 1 doubles the implementation surface of the three most central modules in this project, right when the goal is a deployed, evaluated system as fast as possible — that's the same trap as the original ingestion-perfectionism problem, just spread across two backends instead of one. Shipping the `plain` version end-to-end first, with real eval numbers, guarantees a finished project exists regardless of what happens with the comparison. The comparison itself is genuinely valuable — few candidates run the same golden set through a hand-rolled pipeline and a framework-based one and report the difference — it's just sequenced as a deliberate stretch goal, not a parallel obligation.

**One thing to expect, not fix:** the `langchain` backend will likely score worse on the readability/debuggability principle (rule 6 above) than `plain` — that's an accurate finding about the framework for this use case, not a sign it was implemented badly. Report that as a result; don't paper over it to make the comparison look more balanced than it is.

---

## Phase 0 — Scope
**Time-box: 1–2 days**

Tasks:
- [ ] Confirm domain (default above, or swap)
- [ ] Download the source documents below into `data/raw/`, organized by type (e.g. `data/raw/regulation/`, `data/raw/guidance/`, `data/raw/adjacent/`)
- [ ] Set up repo structure above; `git init`; `docker-compose.yml` with Qdrant + Langfuse services
- [ ] Write a one-page problem statement as the README stub (what this system does, for whom, why the domain was chosen)

### Phase 0 document list (default domain)

| # | Document | Why it's in the corpus | Where to get it | Format |
|---|---|---|---|---|
| 1 | AI Act — Regulation (EU) 2024/1689 (Articles, Recitals, 13 Annexes) | Core regulation, primary source | Official: https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng · Easier to parse: https://artificialintelligenceact.eu/the-act/ (per-Article HTML) | HTML/PDF |
| 2 | Digital Omnibus on AI — Regulation (EU) 2026/1744 | Amends the AI Act's timeline (in force since 27 July 2026) — the versioning/temporal test case | https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng | HTML/PDF |
| 3 | General-Purpose AI Code of Practice (3 chapters: Transparency, Copyright, Safety & Security) | Voluntary but AI-Office-endorsed compliance route, mapped to Articles 53/55 | https://digital-strategy.ec.europa.eu/en/policies/contents-code-gpai | PDF |
| 4 | Guidelines on the scope of obligations for GPAI providers | Commission's interpretive guidance on what counts as a GPAI model | https://digital-strategy.ec.europa.eu/en/policies/guidelines-gpai-providers | PDF |
| 5 | Guidelines on Transparency Obligations (Article 50), finalized 20 July 2026 | Reference document regulators use to assess Article 50 compliance | https://digital-strategy.ec.europa.eu/en/policies/guidelines-ai-transparency-obligations | PDF |
| 6 | Code of Practice on Transparency of AI-Generated Content | Companion to #5 — marking/labelling of deepfakes and AI content | https://digital-strategy.ec.europa.eu/en/policies/code-practice-ai-generated-content | PDF |
| 7 | AI Act Service Desk — per-Article pages (official) | Structured official reference; also useful to spot-check your own extraction against ground truth | https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50 (swap the Article number in the path) | HTML |
| 8 | GDPR — Regulation (EU) 2016/679 | Cross-regulation document, for "does this AI system also trigger GDPR" questions | Official: https://eur-lex.europa.eu/eli/reg/2016/679/oj/eng · Easier to parse: https://gdpr-info.eu/ | HTML/PDF |

*Notes:* EUR-Lex is authoritative but messier to parse than the community/dev-friendly mirrors (artificialintelligenceact.eu, gdpr-info.eu), which preserve clean per-Article heading structure — use EUR-Lex as the citation-of-record link in the UI, and ingest from the cleaner mirrors if extraction quality becomes a fight. Items 3–6 were finalized within weeks of each other in mid-2026 — treat that as real versioning material to test against, not noise to clean up.

**Exit criteria:** repo scaffolded, all 8 documents in place under `data/raw/`, README stub committed. Nothing else.

---

## Phase 1 — Walking skeleton
**Time-box: 3–5 days**
**Goal: the dumbest complete pipeline, deployed. Ugly is fine. Incomplete is not.**

Tasks:
- [ ] Define `app/core/interfaces.py` (Ingestor, Retriever, Generator contracts) — implement everything below against it, in `app/pipelines/plain/`
- [ ] Naive parsing: plain text extraction (HTML→text for the EUR-Lex/AI-Act-EU sources, PyMuPDF for the PDF guidance docs), no special-casing per document yet
- [ ] Fixed-size chunking (e.g. 500 tokens, 50 overlap) — no cleverness
- [ ] Embed chunks, store in Qdrant
- [ ] One retrieval function: top-k dense search only, no reranking, no hybrid
- [ ] One generation call: retrieved context → answer, with source citations in the output
- [ ] FastAPI endpoint wrapping the above
- [ ] Minimal Streamlit UI hitting the endpoint
- [ ] Deploy to Render or Railway

**Explicitly do NOT do in this phase:** OCR, table handling, multi-column logic, per-document-type branching. If extraction looks bad on some pages, that's expected — note it in `PROGRESS.md` and move on.

**Exit criteria:** a live URL exists; you can type a question and get a cited answer, however mediocre.

---

## Phase 2 — Eval harness
**Time-box: 3–4 days**
**Goal: build the thing that got skipped last time — before touching quality.**

Tasks:
- [ ] Draft 30–50 question/answer pairs against the source documents. The agent may draft candidates by reading the documents, but a human must verify every single one before it's added — do not trust auto-generated ground truth.
- [ ] Save as `eval/golden_set.jsonl`, one JSON object per line. Include several questions that specifically test the Regulation-vs-Omnibus versioning conflict (example 2 below) — a stale answer there is the clearest possible demonstration of a real failure mode:
  ```json
  {
    "question": "When do Annex III high-risk AI system obligations become applicable?",
    "expected_answer": "2 December 2027, per the Digital Omnibus amendment — not the original 2 August 2026 date in the base Regulation",
    "expected_source_doc": "digital_omnibus_2026_1744",
    "expected_source_section": "Amendment to Article 113 timeline",
    "difficulty": "temporal-conflict"
  }
  ```
  ```json
  {
    "question": "What must a GPAI provider include in the training data summary under the Code of Practice?",
    "expected_answer": "...",
    "expected_source_doc": "gpai_code_of_practice_transparency",
    "expected_source_section": "Transparency chapter",
    "difficulty": "single-hop"
  }
  ```
- [ ] Wire up DeepEval: write real pytest test functions in `eval/run_eval.py` using `assert_test()` with metric thresholds for faithfulness, answer relevancy, contextual precision, and contextual recall against `golden_set.jsonl`
- [ ] Run against the Phase 1 skeleton, record baseline numbers in `eval/results.md`
- [ ] Add `.github/workflows/eval.yml` running `deepeval test run` on every push, failing the build on any metric threshold breach

**Explicitly do NOT do in this phase:** improve retrieval or generation quality. This phase produces a *number*, not a better answer.

**Exit criteria:** baseline eval numbers exist, committed, and CI runs them automatically.

---

## Phase 3 — Retrieval & generation quality
**Time-box: 1–2 weeks**
**Goal: iterate against the eval harness. Keep only what moves the number.**

Try these one at a time, each measured against the current best score, not just the original baseline:
- [ ] Contextual chunk headers (LLM-generated context prepended to each chunk before embedding)
- [ ] Hybrid search (dense + BM25/sparse, via Qdrant's native support)
- [ ] Reranking (cross-encoder or a hosted reranker)
- [ ] Chunk size / overlap tuning

For each: record the before/after score in `eval/results.md`. Keep the change only if it helps. Log it in `DECISIONS.md` either way — a documented negative result ("tried X, faithfulness dropped because Y, reverted") is worth as much as a positive one for this project's purpose.

**Explicitly do NOT do in this phase:** touch ingestion/parsing sophistication, even if extraction issues are visibly the bottleneck. Note them in `PROGRESS.md` for Phase 4 instead.

**Exit criteria:** eval numbers meaningfully improved over the Phase 2 baseline; a results table exists; at least one negative result is documented.

---

## Phase 4 — Ingestion sophistication (only what the eval proved is broken)
**Time-box: whatever's left — this phase does not get to run long**

Tasks (pursue only the ones Phase 3's eval/manual review actually surfaced as failures):
- [ ] If the Regulation-vs-Omnibus temporal-conflict questions aren't resolving correctly: add effective-date metadata per chunk and prefer the amended value at retrieval/generation time — this is the highest-value fix in this phase for this domain
- [ ] If cross-reference resolution is failing (e.g. a Guideline citing "Article 50" should retrieve the actual Article 50 text): add explicit cross-reference metadata linking chunks across documents
- [ ] If any of the PDF guidance documents (items 3–6 in Phase 0) are extracting poorly: add a Docling-based path for those specifically, gated behind a simple document-type check — not a general parser rewrite
- [ ] Add lightweight confidence-gated routing (fingerprint the document → known recipe vs. escalate) scoped to the document shapes actually present in this corpus — do not build a general-purpose classifier for hypothetical document types you don't have

**Exit criteria:** re-run the Phase 2/3 eval set, confirm improvement specifically on the cases targeted, numbers updated in `eval/results.md`.

---

## Phase 5 — Productionize
**Time-box: 1 week**

Tasks:
- [ ] Langfuse tracing on every LLM call (prompts, latency, cost, token usage)
- [ ] Basic prompt-injection guard: flag/strip suspicious embedded instructions in extracted document text before it reaches the generation prompt; add a test case for it
- [ ] Unit tests: chunking logic, retrieval function, API endpoints
- [ ] README: architecture diagram (Mermaid is fine), results table, "known limitations / failure modes" section, "how to run" section
- [ ] Finalize `DECISIONS.md` so it reads as a coherent narrative, not a raw log
- [ ] Optional: a 2–3 minute recorded walkthrough

**Exit criteria:** every box in "Definition of done" above is checked (excluding the optional Phase 6 line).

---

## Phase 6 — LangChain equivalent (optional stretch, only after Phase 5 is fully done)
**Time-box: 3–5 days**
**Prerequisite: every non-optional box in "Definition of done" is already checked, using the `plain` backend. Do not start this phase as a way to avoid finishing Phase 5.**

Tasks:
- [ ] Implement `app/pipelines/langchain/{ingestion,retrieval,generation}.py` against the same `app/core/interfaces.py` contract used by `plain/`
- [ ] Run the exact same `eval/golden_set.jsonl` through both backends via DeepEval
- [ ] Produce `eval/backend_comparison.md`: metric-by-metric results for both backends, plus a short qualitative section (lines of code, debugging experience, cold-start/latency, what abstractions LangChain bought you vs. cost you for this specific pipeline)
- [ ] Add one paragraph to the README summarizing the comparison and which backend you'd actually pick for this use case, and why

**Exit criteria:** both backends pass through the same eval suite, `backend_comparison.md` exists with real numbers, `PIPELINE_BACKEND` switches between them with no other code changes.

---

## Appendix — explicitly out of scope for v1

- Multi-tenant auth / user accounts
- Streaming token-by-token responses
- Fine-tuning any model
- Supporting file formats beyond PDF
- Horizontal scaling / distributed deployment

---

## Suggested prompt pattern for each Claude Code session

```
Read RAG_PORTFOLIO_SPEC.md in full.
We are starting Phase <N>.
Confirm you understand the exit criteria for this phase before writing any code.
Do not begin Phase <N+1> tasks without my explicit go-ahead, even if it seems efficient to continue.
```
