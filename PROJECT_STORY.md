# Project story — EU AI Act Compliance Assistant

A step-by-step narrative of what was built, why, and what actually happened
when it was tested — written for talking through in an interview, not as a
replacement for `DECISIONS.md` / `PROGRESS.md` / `EVALUATION_HISTORY.md`
(the full detail, numbers, and raw write-ups live there).

**One-line pitch:** a RAG assistant that answers EU AI Act compliance
questions, grounded across 8 cross-referencing regulatory documents,
built through gated phases specifically so evaluation happens early and
often instead of over-investing in ingestion and never measuring anything.

---

## Phase 0 — Scope & data corpus

1. **Domain: EU AI Act compliance.** Chosen over generic alternatives
   because it's current and has real teeth (GPAI transparency enforcement
   began Aug 2026, the "Digital Omnibus" amendment restructured the
   high-risk compliance timeline in July 2026), spans multiple
   cross-referencing documents (not one flat corpus), and has a genuine,
   checkable **versioning test case**: does the system answer with the
   *amended* date or the *original* Regulation's date?
2. **Corpus:** AI Act (Reg. 2024/1689), Digital Omnibus (Reg. 2026/1744),
   GDPR (Reg. 2016/679, cross-reference document), GPAI Code of Practice,
   GPAI-scope guidelines, Article 50 transparency guidelines,
   AI-generated-content Code of Practice, and a curated subset of AI Act
   Service Desk per-Article pages.
3. **Why EUR-Lex via Wayback Machine, not a live fetch:** live EUR-Lex sits
   behind a bot-challenge WAF (also blocked by org browsing policy) — pulled
   the 3 EUR-Lex documents from Internet Archive snapshots instead, verified
   title/article-count against expectations, kept the real EUR-Lex URL as
   the citation-of-record link.

## Phase 1 — Walking skeleton

1. **Chunking:** fixed-size, 500 tokens / 50 overlap (tiktoken `cl100k_base`)
   — the naive baseline, deliberately simple to get something end-to-end
   fast.
2. **Embedding:** local `BAAI/bge-small-en-v1.5` (sentence-transformers), no
   API key needed, behind an `Embedder` protocol so swapping to a hosted
   provider later is a config change, not a rewrite.
3. **Vector store: Qdrant**, embedded/on-disk mode (`qdrant_local_data/`) —
   no Docker needed for local dev; same client API as a real server, so
   pointing at Qdrant Cloud later is a one-line env var (`QDRANT_URL`)
   change, not a code change. Chosen over the alternatives for native
   hybrid (dense + sparse) support, a generous free-tier cloud option, and
   an embedded mode that needs zero infra to start.
4. **Generation:** Groq-hosted `openai/gpt-oss-20b` — fast, cheap, cites
   `source_doc` per chunk used.
5. **FastAPI `/query` + `/health`, minimal Streamlit UI** — tested live in
   browser.
6. **Result:** 837 chunks indexed; manually verified against the spec's own
   temporal-conflict example ("When do Annex III obligations become
   applicable?") — correctly answered **2 December 2027** (the
   Omnibus-amended date), citing both the AI Act and the Digital Omnibus
   together. Qualitative spot-check only — Phase 2 builds the real harness.

## Phase 2 — Eval harness v1

1. **Golden set:** 43 candidate questions drafted (6 hand-verified against
   primary source text, 37 agent-drafted), user reviewed and **rejected 2**
   — final set: 41 questions, `eval/golden_set.jsonl`.
2. **Judge model:** DeepEval's default judge needs an OpenAI key we didn't
   have; reusing the same model that generates answers risks self-grading
   bias — used the larger `gpt-oss-120b` via the existing Groq key as an
   independent-enough judge instead (`eval/judge_model.py`).
3. **Trade-off accepted:** generator and judge are still both GPT-OSS-family
   models from the same lab — real, if reduced, correlated-bias risk, noted
   for the record rather than solved.

---

## Phase 3 — Iterative evaluation & improvement

Everything below is numbered exactly as in `EVALUATION_HISTORY.md` /
`DECISIONS.md` ("Step N"), so this doc cross-references cleanly with the
full write-ups.

### Step 1 — Fix empty Groq responses (prerequisite, not a quality change)
`gpt-oss` models spend part of their token budget on hidden reasoning
tokens; some prompts exhausted the whole 2048-token budget before any
visible answer came out (`finish_reason="length"`). **Fix:**
`max_completion_tokens=4096` + `reasoning_effort="low"` on every Groq call,
and a visible `RuntimeError` instead of silently returning empty text.
Needed before any dense-vs-hybrid comparison could be trusted (different
runs were losing different questions to this crash).

### Step 2 — Hybrid search (dense + BM25 sparse, RRF fusion) — KEPT
**Why:** dense embeddings match on topic similarity but can miss exact-term
matches (article numbers, defined terms) that BM25 catches directly.
**Result:** overall pass rate **27/41 (66%) → 31/41 (76%)**. Kept as default.

### Step 3 — Cross-encoder reranking on top of hybrid — TRIED, REVERTED
Added a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranking pass over the
hybrid candidates. **Result: pass rate 31/41 (76%) → 26/41 (63%)** — a real
regression, especially on multi-hop questions (precision 0.844→0.602).
**Why reverted:** the reranker is trained on general web passage ranking,
not legal text, and appears to work against hybrid's own
semantic+exact-term balance rather than refine it. Code kept, disabled by
default (`USE_RERANKING=false`).

### Step 4 — Eval harness extended: component + application-level checks
Found the harness only checked answer-vs-context quality, nothing checked
retrieval correctness independently or whether the answer was simply
*correct*. Added a reference-based **retrieval-hit** check (no LLM call) and
a DeepEval **Answer Correctness** check. **Finding:** retrieval-hit was only
80% (33/41) — lower than the LLM-judged metrics implied, because those can
score well even when a *different* chunk than expected happens to contain
equivalent info. New combined baseline: **24/41 (59%)** across all 6 checks.

### Step 5 — Chunking-strategy screening (Groq-free)
Recursive chunking (natural paragraph/sentence boundaries instead of a hard
token cut) screened at **34/41 (83%)** retrieval-hit vs. fixed's 33/41
(80%) — a promising but not yet judged candidate.

### Step 6 — First external judge run (Claude Sonnet 5) on hybrid + recursive
Started using Claude Sonnet 5 as an independent judge (reduces the
same-lab self-grading-bias risk of Groq judging Groq) — the beginning of the
**external-judging pattern**: export a JSON file + self-contained
instructions, user runs it through a separate Sonnet-5 session, results come
back and get folded in. **Standing rule established from here on: Groq is
for answer generation only, never judging or any other LLM-assist task**
(contextual headers, decomposition, etc. all go through this same export
pattern). **Result:** 82.9% doc-recall (confirms Step 5's screen), 71% of
answers fully/mostly correct. Found: citations just echo the retrieved set
rather than reflecting real support; document-level hit-rate hides
chunk-level misses; one hallucination under good retrieval; a
self-consistency failure across two near-duplicate Article 53(2) questions.

### Step 7 — Contextual chunk headers — KEPT, new best
Per Anthropic's "contextual retrieval" technique: exported every chunk +
its document, had Sonnet-5 write a 1-2 sentence situating header per chunk
externally, prepended each header to its chunk before embedding.
**Result: 36/41 (87.8%)** retrieval-hit, beating recursive chunking's 83%.
Not yet combined with recursive chunking at this point (different chunk
boundaries, would need regenerating).

### Step 8 — Hard-subset judging (15 targeted questions)
Instead of re-judging all 41, judged the 15 structurally hardest (8
cross-reference, 3 multi-hop, 2 temporal-conflict, 2 still-missing
single-hop) — cheaper judging effort spent where failures actually cluster.
**Result:** 67% doc-recall / 53% fully-or-mostly-correct (expected lower —
deliberately the hard tail). **New findings:** doc-path scoring has false
negatives too, not just false positives; GDPR retrieval fails specifically
on *scenario-phrased* questions, not mechanism-named ones; found the most
consequential error of any judging round — a confident wrong citation,
**Article 54(6) cited in place of 53(2)** (this exact bug resurfaces
independently in Step 18, ten steps later).

### Step 9 — Diagnosis: why is GDPR retrieval inconsistent? (no config change)
Pure retrieval-mechanics investigation, no LLM calls. **Three distinct root
causes**, not one: (1) relevant chunk found but ranked just outside the
top-5 cutoff after RRF fusion; (2) genuine semantic mismatch — a
scenario-phrased business question doesn't embed close to formal statutory
phrasing, even at very wide recall depth (relevant chunk ranked 65th-418th
out of 837); (3) a compound dual-regulation query gets dominated by
whichever regulation's vocabulary is more prominent in the question text
(GDPR chunk invisible at rank 293-336 because the query "sounds like"
AI-Act language). **Key insight:** the real dividing line isn't GDPR-vs-AI-Act
or cross-reference-vs-single-hop — it's whether the question names the
specific legal mechanism, or only describes a scenario needing translation
to one.

### Step 10 — Full 41-question external judging
Extended Step 8's targeted judging to all 41 for a complete picture.
**Result:** 87.8% doc-recall (exact match with Step 7's screen), 82.9%
fully/mostly correct — best full-set result yet. Also found: 2 Article 50
questions both miss the same specific passage (paragraph 38) — a likely
chunk-boundary issue, not yet fixed.

### Step 11 — GDPR cross-reference boost (keyword-triggered) — KEPT
Considered 3 fixes for Step 9's diagnosis: (a) raise `RETRIEVAL_TOP_K`
globally, (b) query reformulation/HyDE, (c) a metadata-filtered secondary
search restricted to GDPR when the question names it. **Chose (c)** — zero
LLM calls, cheap, validated before implementing.
**Why not (b):** needs a live LLM call per query at retrieval time —
against the "Groq for generation only" rule from Step 6 (this exact
rejection gets revisited and reversed later, at Step 24).
**Result:** retrieval-hit 87.8% → **95.1%**, all 3 failing GDPR questions
fixed, zero regressions.

### Step 12 — Corpus-side cross-reference tagging — KEPT, generalizes Step 11
Step 11 only fired on a literal "GDPR" keyword in the question — wouldn't
generalize to phrasing that implies GDPR without naming it. Added a second
signal: at ingestion, tag every chunk with which other documents it
explicitly names (`references` payload field); at retrieval, the boost also
fires when an *already-retrieved chunk* names an unrepresented document —
reacting to what the documents say, not how the question is phrased.
**Result: 95.1% → 97.6%** — a second, previously-failing, non-GDPR question
also got fixed as an unprompted side effect, real evidence it generalizes.

### Step 13 — Judged the boost's actual cost: a real, mixed result
Before this step, the boost was kept on retrieval-hit numbers alone.
Judging the 15 affected questions found: Retrieval Hit and Chunk-Level Hit
both improved sharply, but **Answer Correctness (0.803→0.760) and Citation
Accuracy (0.617→0.510) both got worse**. One severe regression: a
correct-if-incomplete answer became a **non-answer** once 5 irrelevant GDPR
chunks were added — the model had the right evidence both times but talked
itself out of using it once buried in noise. **Lesson used going forward:**
measure the downstream cost before assuming a retrieval win is a net win.

### Step 14 — Real qrels file: true Precision/Recall, for free, forever
Realized none of the metrics so far measured *how much of what's retrieved
is actually relevant* — only "is at least one relevant chunk present."
Built `eval/qrels.jsonl` via pooling (union of top-20 from 4 independent
retrieval methods, judged by Sonnet-5 for genuine relevance, not topical
proximity). **New finding:** even the best config only reaches **0.356
precision** — about two-thirds of everything retrieved isn't actually
relevant. Retrieval-hit-style metrics could never have surfaced this.

### Step 15 — CI bug: contextual headers never enabled in CI
CI's retrieval-hit check started failing at 68.3% (28/41). Root cause:
`USE_CONTEXTUAL_HEADERS` was never set in the CI workflow's ingestion step —
CI had been silently testing a headerless corpus against a 93%-floor gate
that assumed headers were on. **Fix:** one env var added to `eval.yml`.

### Step 16 — The *real* remaining CI cause: 6 PDFs never committed to git
After Step 15's fix, CI still failed at 70.7%. First hypothesis (PDF
extraction differs Linux vs. Windows) was **checked, not assumed** —
diffing PDF extraction output byte-for-byte across versions ruled it out,
and reproducing the exact CI config locally hit 97.6%, not CI's number. A
temporary diagnostic step revealed the real cause: `.gitignore` excluded
`data/raw/**/*.pdf`, and no re-fetch script existed — **the 6 guidance PDFs
were simply never in the repo.** Would have hit production identically on
any fresh deploy checkout. **Fix:** committed the PDFs directly (~4.85MB).
**Process point:** the first plausible-sounding explanation would have
been wrong if shipped without the diagnostic step to falsify it first.

### Step 17 — Three named, permanent corpora for side-by-side comparison
Built `ai_act_corpus_fixed_hybrid`, `ai_act_corpus_recursive_hybrid`, and
`ai_act_corpus_recursive_ctxheaders_hybrid` as permanent Qdrant collections
alongside production, so the dev UI can compare configs interactively
instead of only through one-off scripts.

### Step 18 — Recursive chunking regresses on a citation; root cause chased down
Manually ran one hard multi-hop question through all 4 corpora. Both fixed
corpora correctly cited **Article 53(2)**; recursive-only cited the wrong
sub-paragraph; **recursive+headers cited Article 54(6) — a wrong article
entirely**, the *exact same hallucination* Step 8 found independently, ten
steps earlier, on a different corpus build. Three diagnostics chased the
actual mechanism (ruling out header-projection compounding and generic
"chunk cleanliness" along the way): **the real cause is RRF fusion
structurally rewarding a chunk that ranks decently on *both* signals over
one that ranks #1 on a single signal but is invisible on the other.** The
correct chunk (`article_53.html#0`) had the single best dense score of any
candidate but zero BM25/keyword overlap; the wrong chunk
(`article_55.html#9`) wasn't best on either signal alone but was solidly
good on both — RRF fusion favored the "generalist" over the "specialist."
Confirmed as a real, reproducible finding, not a fluke — but scoped to one
question, not yet shown to be systematic.

### Step 19 — Query decomposition fixes the exact Step 18 bug (offline test)
Hypothesis: retrieving once per sub-question instead of once for the whole
compound question should stop one concept's chunk from "diluting" or
getting outcompeted by another concept's. Sub-questions generated externally
by Sonnet-5 (decomposition needs an LLM call — same "Groq for generation
only" rule conflict as Step 11's HyDE rejection, resolved via the same
export pattern). **Result:** fixed the exact Article 53/54 citation bug on
the corpus where it was found — precision on that question jumped
0.30→0.86. **Not universal:** one other question lost precision (more noise
chunks, no correctness change); decomposition retrieves noticeably more
chunks per run with no cap in place.

### Step 20 — Qrels made chunking-agnostic; recursive+headers now best on P/R
Step 14's qrels file was chunk-index-based, meaningless once recursive
chunking produces different chunk boundaries. Migrated to
character-span-based relevance judgments (pure re-representation, no new
LLM judging) so Precision/Recall works for any chunking strategy.
**Headline result, genuinely in tension with Step 18:** recursive+headers
is now the **best-scoring corpus of all four on real Precision/Recall**
(P 0.41 / R 0.74) — an aggregate win and a real single-question regression
are simultaneously true; they answer different questions. Re-ran Step 19's
decomposition test with the fix: the regression question jumped even
further, 0.30/0.67 → **0.86/1.00**.

### Step 21 — Full external judging of recursive+headers: beats production
Before promoting anything, ran the full 41-question Sonnet-5 judging
rigor on recursive+headers. **Beat old production on every axis:**
retrieval hit 100% vs. 87.8%, faithfulness 0.995 vs. 0.909, answer
correctness 0.934 vs. 0.828, citation accuracy 0.755 vs. 0.663. Critically
checked, not just celebrated: the Article 53/54 question scored "Correct"
this round, but its retrieved chunks were byte-identical to Step 18/20's
buggy runs — meaning this was generation-noise landing correctly once, not
the retrieval risk being fixed. The bug was confirmed **non-deterministic**,
not resolved.

### Step 22 — Recursive+headers promoted to production
Given Step 20 (best P/R) and Step 21 (best judged quality, stable under
extra judge thinking effort), **explicitly chose to switch now, accepting
the known, latent, intermittent citation risk** from Step 18 rather than
wait to wire in decomposition first. Pure env-var switch
(`QDRANT_COLLECTION`, `CHUNKING_STRATEGY`, `USE_CONTEXTUAL_HEADERS`,
`CONTEXTUAL_HEADERS_PATH`) — old collection untouched, rollback is reverting
4 env vars, no re-ingestion needed.

### Step 23 — Chain-of-thought prompting alone does NOT fix the citation bug
Tested chain-of-thought generation (reason under a `Reasoning:` section
before an `Answer:` section — same Groq call, no new LLM call) combined
with decomposition, 4 variants (`baseline`/`cot`/`decomposed`/
`decomposed_cot`). Qdrant Cloud connectivity was down, so built a local,
on-disk mirror of production config to test against instead of waiting.
**Result:** on the regression question, `baseline` and `cot` both still
cited "Article 54(6)" (wrong); `decomposed` and `decomposed_cot` both
correctly cited "Article 53(2)". **CoT changed phrasing/confidence on every
question but never changed which sources got cited on any of them** —
confirms the fix has to happen in *what gets retrieved*, not in how the
model reasons over whatever it's handed.

### Step 24 — Query decomposition wired into the live pipeline via Groq
Explicit, deliberate reversal of the "no live LLM call at retrieval time"
rule from Step 11/19 — weighed the cost first (~1.8-2.4x per-question
generation cost from larger merged context; the decomposition call itself
near-free, ~$0.00006/question), then wired it in on purpose. One Groq call
per question now decides *whether* it's multi-hop and splits it if so, in a
single call — a single-hop question pays one cheap extra call and retrieves
exactly as before; a multi-hop one retrieves per sub-question and merges.
**Fails open:** any error falls back to the original, undecomposed question
rather than breaking the answer. **Verified:** live decomposer calls
correctly returned 1 sub-question for single-hop and 2-3 for known
multi-hop questions; full retrieve→generate path re-ran the exact Step 18
regression question end-to-end and correctly cited "Article 53(2)".

---

## Where this stands now

- **Live production config:** recursive chunking + projected contextual
  headers + hybrid retrieval + cross-reference boost + query decomposition,
  all via env vars, old collection kept for instant rollback.
- **Known open risk:** query decomposition mitigates but wasn't proven to
  eliminate every instance of the RRF generalist-vs-specialist failure mode
  — only directly confirmed fixed on the one question it was found on.
- **Not yet re-verified against the live Qdrant Cloud collection** — recent
  work (Steps 23-24) validated against a local, on-disk mirror while Cloud
  connectivity was down on this machine; re-run once that's confirmed
  restored.
- **Recurring theme worth telling in an interview:** every major win came
  with a matching negative result or reverted attempt right next to it
  (reranking reverted, cross-reference boost's real judged cost, CI bugs
  chased down with disproven-first hypotheses, decomposition's non-universal
  benefit) — the discipline was measuring before and after every change,
  not just keeping whichever number looked best.
