# Task: Decompose multi-hop questions into independent sub-questions

## Context

A retrieval-augmented question-answering system answers questions about
obligations under the EU AI Act and related regulations, by retrieving small
text chunks (roughly 500 tokens each) from a fixed corpus of source documents
and using them to generate a grounded, cited answer. The corpus includes: the
AI Act itself (Regulation (EU) 2024/1689), the "Digital Omnibus" amendment
(Regulation (EU) 2026/1744) which changed several of the AI Act's compliance
dates, GDPR (Regulation (EU) 2016/679), and several pieces of Commission/AI
Office guidance (Codes of Practice, scope guidelines, transparency
guidelines) plus an "AI Act Service Desk" set of per-Article summary pages.

**The problem being tested:** some questions require pulling together
information from *two or more distinct provisions or documents at once*
("multi-hop" questions). Retrieving with the question exactly as asked can
retrieve chunks for only one of the underlying concepts, or a single
embedding for a compound question can dilute across multiple concerns and
retrieve a chunk that's a decent semantic match for the *whole* question but
not the specific right provision. The hypothesis being tested is that
breaking such a question into independent, narrower sub-questions — each
answerable by retrieving one focused chunk — and retrieving separately for
each, then merging the results, may retrieve more precisely than retrieving
once for the combined question.

## What to do

**Input:** `eval/questions_for_decomposition.jsonl` — one JSON object per
line:

```json
{"question": "..."}
```

3 questions total, all confirmed multi-hop (each genuinely requires
information from two or more distinct provisions/documents to answer fully).

For each question, write **2-4 independent sub-questions** such that:

- Each sub-question is **self-contained** — it must not refer back to the
  original question or to another sub-question (no "it", "this", "the first
  one" — restate whatever context is needed in full).
- Each sub-question targets **one specific, narrow piece of information**
  that could plausibly be answered by retrieving a single focused chunk of
  regulatory text — not a second compound question.
- Together, the sub-questions' answers should cover everything needed to
  fully answer the original question. Don't over-split a question that's
  only borderline multi-hop into artificially many pieces — 2 sub-questions
  is fine if that's genuinely all the original question needs; use 3-4 only
  when the question truly has that many distinct parts.
- Preserve the original question's specific legal terminology, article
  numbers, and document names exactly as given — don't paraphrase them away
  or generalize them, since the retrieval system matches on this specific
  language.

**Good example** (illustrative, not one of the 3 actual input questions):

> Original: "Does the Digital Omnibus change how the AI Act treats
> high-risk system documentation, and if so, does GDPR impose any
> additional documentation requirement on top of that?"
>
> Sub-questions:
> 1. "Does the Digital Omnibus (Regulation (EU) 2026/1744) change the AI
>    Act's documentation requirements for high-risk AI systems, and if so,
>    how?"
> 2. "Does GDPR (Regulation (EU) 2016/679) impose any documentation or
>    record-keeping requirement that would apply on top of the AI Act's
>    high-risk system documentation obligations?"

**Bad sub-questions:** "What does the Digital Omnibus say?" (too vague, not
targeted); "What about GDPR?" (not self-contained, no specific ask); a
sub-question that's really just the original question restated once more
(not actually decomposed).

## Output format

One JSONL file, one line per original question, in the same order as the
input:

```json
{"question": "<original question, copied exactly>", "sub_questions": ["<sub-question 1>", "<sub-question 2>"]}
```

Save as `decomposed_questions.jsonl` and return it.
