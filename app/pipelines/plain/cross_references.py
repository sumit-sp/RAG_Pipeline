"""Corpus-side cross-reference detection (Phase 3, Step 12 — generalizes the
GDPR-only keyword boost from Step 11, see EVALUATION_HISTORY.md).

Legal texts routinely name other regulations directly ("without prejudice to
Regulation (EU) 2016/679", "as amended by the Digital Omnibus"). Scanning for
these mentions lets retrieval react to what the documents themselves say a
chunk relates to, instead of guessing from the user's question wording —
`detect_references` is used both at ingestion time (to tag each chunk with
what it names) and at retrieval time (on the question text itself, as a
cheap first signal alongside the chunk-tag one — see retrieval.py).

Deliberately conservative: only documents with a distinctive, unambiguous
identifying phrase (an official regulation number, not just a colloquial
name) are registered, so a match is never a guess. The GPAI Code of
Practice's three chapters, for instance, share the same generic name and
aren't registered here — there'd be no reliable way to tell from "GPAI Code
of Practice" alone which of the three separate PDFs is meant.
"""

# alias (matched case-insensitively, as a plain substring) -> the source_doc
# it unambiguously identifies.
_REFERENCE_ALIASES: dict[str, str] = {
    "gdpr": "adjacent/gdpr_2016_679.html",
    "general data protection regulation": "adjacent/gdpr_2016_679.html",
    "regulation (eu) 2016/679": "adjacent/gdpr_2016_679.html",
    "regulation 2016/679": "adjacent/gdpr_2016_679.html",
    "digital omnibus": "regulation/digital_omnibus_2026_1744.html",
    "regulation (eu) 2026/1744": "regulation/digital_omnibus_2026_1744.html",
    "regulation 2026/1744": "regulation/digital_omnibus_2026_1744.html",
    "regulation (eu) 2024/1689": "regulation/ai_act_2024_1689.html",
    "regulation 2024/1689": "regulation/ai_act_2024_1689.html",
}


def detect_references(text: str, exclude_source_doc: str | None = None) -> list[str]:
    """Returns the sorted, de-duplicated source_docs this text names by alias.

    `exclude_source_doc` drops a chunk's own document from its own results
    (so a GDPR chunk that itself says "GDPR" doesn't tag itself as
    referencing GDPR) — pass nothing when scanning a question, since a
    question has no source_doc of its own.
    """
    text_lower = text.lower()
    found = {
        source_doc
        for alias, source_doc in _REFERENCE_ALIASES.items()
        if alias in text_lower and source_doc != exclude_source_doc
    }
    return sorted(found)
