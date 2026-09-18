"""Phase 1 generation: retrieved context -> answer, with source citations."""

from groq import Groq

from app.core import config
from app.core.models import Answer, RetrievedContext

_SYSTEM_PROMPT = (
    "You are a compliance assistant answering questions about the EU AI Act and "
    "related regulations (Digital Omnibus, GPAI guidance, GDPR). Answer using ONLY "
    "the provided context excerpts — do not use outside knowledge. If the context "
    "doesn't contain the answer, say so plainly instead of guessing."
)


def _format_context(contexts: list[RetrievedContext]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source_doc: {c.chunk.source_doc})\n{c.chunk.text}"
        for i, c in enumerate(contexts)
    )


class PlainGenerator:
    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)

    def generate(self, question: str, contexts: list[RetrievedContext]) -> Answer:
        user_prompt = (
            f"Context excerpts:\n{_format_context(contexts)}\n\nQuestion: {question}"
        )
        response = self.client.chat.completions.create(
            model=config.GENERATION_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        text = response.choices[0].message.content

        citations = sorted({c.chunk.source_doc for c in contexts})
        return Answer(text=text, citations=citations)
