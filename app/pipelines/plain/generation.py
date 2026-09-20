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
            max_completion_tokens=4096,
            reasoning_effort="low",
        )
        choice = response.choices[0]
        text = choice.message.content
        if not text:
            # gpt-oss models spend part of the token budget on hidden "reasoning"
            # tokens before the visible answer; with finish_reason="length" that
            # budget can run out before any answer is emitted. Surface this as a
            # visible failure instead of silently returning an empty answer.
            raise RuntimeError(
                f"Groq returned an empty response (finish_reason={choice.finish_reason!r}) "
                f"for question: {question!r}"
            )

        citations = sorted({c.chunk.source_doc for c in contexts})
        return Answer(text=text, citations=citations)
