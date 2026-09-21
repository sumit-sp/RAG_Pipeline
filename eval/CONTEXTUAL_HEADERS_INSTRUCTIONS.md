# Task: Generate contextual chunk headers

**Input:** `eval/chunks_for_contextual_headers.jsonl` — one JSON object per line,
one line per source document:

```json
{
  "source_doc": "regulation/ai_act_2024_1689.html",
  "doc_type": "regulation",
  "full_text": "<the entire extracted document text>",
  "chunks": [
    {"chunk_index": 0, "text": "<chunk 0 text>"},
    {"chunk_index": 1, "text": "<chunk 1 text>"}
  ]
}
```

25 documents, 837 chunks total.

## What to do, per document

Using `full_text` as the document's full context, write a short header (1-2
sentences, roughly 30-60 words) for **every** chunk in that document's `chunks`
list. The header situates the chunk within the document — what document it's
from, and what specific provision/topic this particular chunk covers — so that a
reader (or an embedding model) understands the chunk without needing the
surrounding text.

**Good header example** (for a mid-document chunk with no visible heading of its
own): *"From the EU AI Act (Regulation 2024/1689), Article 113 — this chunk
covers the amended application timeline for high-risk AI systems under Annex
III, as modified by the Digital Omnibus."*

**Bad header:** anything that just restates the chunk's own first sentence, or is
generic enough to apply to any chunk in the document ("This is part of the AI
Act.").

Do not summarize the whole document in every header — each header is specific to
its own chunk's content within the document.

## Output format

One JSONL file, one line per chunk, across all documents:

```json
{"source_doc": "regulation/ai_act_2024_1689.html", "chunk_index": 0, "header": "..."}
{"source_doc": "regulation/ai_act_2024_1689.html", "chunk_index": 1, "header": "..."}
```

Save as `eval/contextual_headers.jsonl` (837 lines expected) and bring it back —
`app/pipelines/plain/contextual_headers.py` already knows how to consume this
exact format at ingestion time.
