"""Chain-of-thought variant of PlainGenerator, for the decomposition+CoT
comparison experiment (eval/decomposition_cot_experiment.py) and for ad-hoc
use from notebooks/rag_experiments.ipynb.

Same Groq call as app/pipelines/plain/generation.py -- generation is the one
LLM-assist task the Groq-restriction rule (see DECISIONS.md) allows, and
prompting the model to reason before answering doesn't add a second call.
The only change is the system prompt and a light parse step to split the
model's visible reasoning from its final answer, so citations/scoring still
operate on the final-answer text alone rather than the reasoning trace.
"""

import re

from groq import Groq

from app.core import config
from app.core.models import Answer, RetrievedContext
from app.core.tracing import trace_generation
from app.pipelines.plain.generation import _format_context

_COT_SYSTEM_PROMPT = (
    "You are a compliance assistant answering questions about the EU AI Act and "
    "related regulations (Digital Omnibus, GPAI guidance, GDPR). Answer using ONLY "
    "the provided context excerpts -- do not use outside knowledge. If the context "
    "doesn't contain the answer, say so plainly instead of guessing.\n\n"
    "Think step by step before answering: identify which excerpts are actually "
    "relevant to the question, note any cross-references between them, and reason "
    "about what they say before committing to a final answer. Structure your "
    "response as exactly two sections, in this order:\n"
    "Reasoning: <your step-by-step reasoning>\n"
    "Answer: <your final answer to the question, standing on its own>"
)

_ANSWER_SPLIT_RE = re.compile(r"answer\s*:\s*", re.IGNORECASE)


def _split_reasoning_and_answer(raw_text: str) -> tuple[str, str]:
    """Returns (reasoning, final_answer). Falls back to (raw_text, raw_text)
    if the model didn't follow the Reasoning/Answer structure, since the
    Groq gpt-oss models are not guaranteed to obey formatting instructions
    exactly and we'd rather score/display *something* than crash."""
    parts = _ANSWER_SPLIT_RE.split(raw_text, maxsplit=1)
    if len(parts) == 2:
        reasoning = re.sub(r"^\s*reasoning\s*:\s*", "", parts[0], flags=re.IGNORECASE).strip()
        # gpt-oss sometimes bolds the section label itself ("**Answer:**"),
        # which leaves a dangling "**" here since the split happens right
        # after the colon -- strip leftover markdown emphasis markers/blank
        # lines rather than surface them as if they were part of the answer.
        final_answer = re.sub(r"^[\s*]+", "", parts[1]).strip()
        return reasoning, final_answer
    return raw_text.strip(), raw_text.strip()


class CotGenerator:
    """Drop-in alternative to PlainGenerator: same interface (.generate),
    same model/client, chain-of-thought system prompt instead."""

    def __init__(self, reasoning_effort: str = "medium"):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.reasoning_effort = reasoning_effort

    def generate(self, question: str, contexts: list[RetrievedContext]) -> Answer:
        user_prompt = (
            f"Context excerpts:\n{_format_context(contexts)}\n\nQuestion: {question}"
        )
        model_parameters = {
            "temperature": 0.0,
            "max_completion_tokens": 4096,
            "reasoning_effort": self.reasoning_effort,
        }
        with trace_generation(
            name="cot-generator",
            input={"system": _COT_SYSTEM_PROMPT, "user": user_prompt},
            model=config.GENERATION_MODEL,
            model_parameters=model_parameters,
        ) as trace:
            response = self.client.chat.completions.create(
                model=config.GENERATION_MODEL,
                messages=[
                    {"role": "system", "content": _COT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                **model_parameters,
            )
            choice = response.choices[0]
            raw_text = choice.message.content
            if not raw_text:
                raise RuntimeError(
                    f"Groq returned an empty response (finish_reason={choice.finish_reason!r}) "
                    f"for question: {question!r}"
                )

            usage = response.usage
            usage_dict = (
                {
                    "input": usage.prompt_tokens,
                    "output": usage.completion_tokens,
                    "total": usage.total_tokens,
                }
                if usage
                else None
            )
            trace.set_output(raw_text, usage=usage_dict)

        reasoning, final_answer = _split_reasoning_and_answer(raw_text)
        citations = sorted({c.chunk.source_doc for c in contexts})
        answer = Answer(text=final_answer, citations=citations, usage=usage_dict)
        answer.reasoning = reasoning  # stashed for notebook inspection; not part of Answer's schema
        answer.raw_text = raw_text
        return answer
