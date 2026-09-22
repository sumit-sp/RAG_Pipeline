# Task: Build a relevance-judgment set (qrels) for a RAG retrieval system

## Background

This is for a retrieval-augmented generation (RAG) system that answers
compliance questions about the EU AI Act (an EU regulation) and related
documents (a related "Digital Omnibus" amendment, GDPR, and various
Commission/industry guidance documents). The system works by first
retrieving a handful of text chunks from a ~800-chunk corpus that seem
relevant to a user's question, then having a language model write an answer
grounded in those chunks.

The problem this task solves: to properly evaluate how good the retrieval
step is, you need to know — for each question — the *complete* set of
chunks that are actually relevant to it, not just one example of a relevant
chunk. Without that, you can't calculate standard retrieval-quality metrics
like Precision (what fraction of what was retrieved is actually relevant) or
Recall (what fraction of everything relevant was actually retrieved). This
task builds that missing ground truth: for each question, you'll read a set
of candidate chunks and mark which ones are genuinely relevant.

**Input:** a JSONL file (one JSON object per line), each line shaped like:

```json
{
  "question": "...",
  "difficulty": "single-hop | cross-reference | multi-hop | temporal-conflict",
  "expected_answer": "<the correct/golden answer to the question>",
  "expected_source_doc": "regulation/ai_act_2024_1689.html",
  "expected_source_section": "<article/section reference, e.g. 'Article 5(1)(h)'>",
  "candidates": [
    {"source_doc": "...", "chunk_index": 0, "text": "<chunk text>"},
    ...
  ]
}
```

Notes on the fields:
- `difficulty` just categorizes the question type; it doesn't affect how you judge relevance.
- `expected_answer` is the correct answer — use it to understand what facts a relevant chunk would need to contain.
- `expected_source_doc` / `expected_source_section` point to one known-good location the answer can be found, but they are **not** the full set of relevant chunks — that's exactly what you're building. Other chunks (possibly from other documents) may also be genuinely relevant.
- Each candidate chunk's `text` sometimes starts with a short one- or two-sentence header (e.g. "From Article 9 of the GDPR, covering...") that was added to situate the chunk before the actual excerpt begins — this is normal, not an error, and can help you understand what the chunk covers.

**Process one input file per session** — each contains 5-6 questions and
roughly 30-40 candidate chunks per question (~200 chunks total). Please read
each candidate chunk carefully rather than skimming; there is no need to
rush, and a smaller, careful batch is exactly why the file was sized this way.

## Where the candidates came from (so you understand the scope)

The `candidates` list for each question is **not** every chunk in the entire
corpus — exhaustively checking all ~800 chunks against every question isn't
practical. Instead, each question's candidate list was built by combining
the top results from several different automated search methods run against
that question (e.g. keyword-based search, semantic/embedding-based search,
and a combined approach), then merging and de-duplicating them into one
pool. The assumption is that a genuinely relevant chunk is very likely to
show up in at least one of those search methods' results, even if it's
ranked low.

**Practical consequence:** it's possible (and fine) for a question to have
very few or even zero truly relevant chunks in its candidate list, if none
of the search methods happened to surface the right content. Just judge what's
actually in front of you — don't assume there must be a "correct" chunk hiding
somewhere in the list if you don't see one.

## What "relevant" means — the most important part of this task

Mark a chunk relevant **only if its content would actually be used or needed
to construct or verify the correct answer** — not just because it's from the
same regulation, mentions the same general topic, or happens to be near the
right article/section.

This distinction matters a lot and is easy to get wrong. A very common
mistake: seeing a chunk that merely *mentions* the right topic in passing
(e.g. a boilerplate legal disclaimer like "this provision applies without
prejudice to [some other regulation]") and marking it relevant just because
it name-drops something related. That is **not** enough — the chunk has to
actually contain the specific fact, date, rule, obligation, or exception that
the question is asking about.

- **Relevant:** a chunk containing the specific fact, date, obligation,
  exception, or rule that the `expected_answer` states — even if worded
  differently than the golden answer, even if it's from a document other
  than `expected_source_doc`, and even if it only covers part of what's
  needed (e.g., one of two facts a multi-part question requires).
- **Not relevant:** a chunk from the right general area of law, the right
  document, or even the right article, that doesn't actually contain
  anything needed to answer *this specific* question — background recitals
  that mention the same regulation in passing, unrelated neighboring
  provisions, generic cross-references, or navigation/table-of-contents-style
  text with no substantive content.
- **Quick test when unsure:** ask yourself, "if this were the *only* chunk
  someone had to answer the question, would it actually let them construct
  or meaningfully support the correct answer?" If the honest answer is no,
  it's not relevant — even if it feels thematically close.

## Output format

For each question, list only the chunks you judged **relevant**. Chunks not
listed are automatically treated as judged "not relevant" — you don't need
to write an entry for every non-relevant chunk in the pool. Output one JSON
object per line, in the same order as the input, one line per question:

```json
{"question": "...", "relevant_chunks": [{"source_doc": "...", "chunk_index": 0, "reason": "states the exact date the question asks about"}, ...]}
```

- `reason` should be a short phrase (a few words) explaining why the chunk qualifies — not a full explanation, just enough for someone to sanity-check your judgment later.
- If a question genuinely has zero relevant chunks in its candidate pool, output `"relevant_chunks": []` for it — don't force a marginal match just to avoid an empty list.

Please name the output file the same as the input file with `_judged` added
before the extension (e.g. `batch_01.jsonl` → `batch_01_judged.jsonl`).
