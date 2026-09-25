"""Live query decomposition (Step 24): one Groq call per question decides
whether it's multi-hop and, if so, splits it into self-contained
sub-questions, retrieved separately and merged by PlainRetriever.retrieve().

Same prompt shape validated offline in eval/DECOMPOSITION_INSTRUCTIONS.md
(Step 19/23: fixed the exact Article 53(2)/54(6) citation regression from
Step 18), condensed into one instruction and combined with the "is this even
multi-hop" decision so a single-hop question costs one cheap extra call and
retrieves exactly as before, instead of always fanning out to N sub-question
retrievals.

This is a deliberate reversal of the "no live LLM call at retrieval time"
rule Steps 11/19 used to justify keeping HyDE/decomposition offline-only --
see DECISIONS.md's Step 24 entry for why. Fails open: any API error or
malformed response falls back to `[question]` (retrieval behaves exactly as
if decomposition were off for that one call) rather than breaking the
user-facing answer over a decomposition hiccup.
"""

import json
import logging

from groq import Groq

from app.core import config
from app.core.tracing import trace_generation

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You help a retrieval system decide how to search a corpus of EU AI Act, "
    "Digital Omnibus, GPAI guidance, and GDPR text for a compliance question.\n\n"
    "If the question can be fully answered by retrieving a single focused "
    "passage, return it unchanged as the only item in sub_questions.\n\n"
    "If the question genuinely requires combining information from two or "
    "more distinct provisions or documents (a \"multi-hop\" question), split "
    "it into 2-4 sub-questions such that each one:\n"
    "- is self-contained (no \"it\"/\"this\"/pronouns referring back to the "
    "original question or another sub-question -- restate whatever context "
    "is needed in full)\n"
    "- targets one specific, narrow piece of information answerable by a "
    "single focused passage of regulatory text, not a second compound "
    "question\n"
    "- preserves the original question's exact legal terminology, article "
    "numbers, and document names\n"
    "Together the sub-questions should cover everything needed to fully "
    "answer the original question. Don't over-split a question that's only "
    "borderline multi-hop.\n\n"
    'Respond with ONLY a JSON object of the form {"sub_questions": ["..."]} '
    "-- no other text."
)


def _parse_sub_questions(raw: str) -> list[str]:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    sub_questions = data.get("sub_questions")
    if not isinstance(sub_questions, list) or not sub_questions:
        return []
    if not all(isinstance(s, str) and s.strip() for s in sub_questions):
        return []
    return [s.strip() for s in sub_questions]


class QueryDecomposer:
    def __init__(self, model: str | None = None):
        self._default_client = Groq(api_key=config.GROQ_API_KEY)
        self.model = model or config.QUERY_DECOMPOSITION_MODEL

    def decompose(self, question: str, client: Groq | None = None) -> list[str]:
        """Returns [question] unchanged for a single-hop question (or on any
        failure), or 2-4 self-contained sub-questions for a multi-hop one.

        `client`, when given, is used for this call only -- see
        PlainGenerator.generate()'s docstring for why (shared-singleton
        safety with a byok-resolved, per-request client)."""
        client = client or self._default_client
        model_parameters = {
            "temperature": 0.0,
            "max_completion_tokens": 512,
            "reasoning_effort": "low",
        }
        try:
            with trace_generation(
                name="query-decomposer",
                input={"system": _SYSTEM_PROMPT, "user": question},
                model=self.model,
                model_parameters=model_parameters,
            ) as trace:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    **model_parameters,
                )
                raw = response.choices[0].message.content or ""
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
                trace.set_output(raw, usage=usage_dict)

            sub_questions = _parse_sub_questions(raw)
        except Exception:
            logger.warning(
                "Query decomposition failed for %r; retrieving the original question unchanged.",
                question,
                exc_info=True,
            )
            return [question]

        if not sub_questions:
            return [question]
        return sub_questions[: config.QUERY_DECOMPOSITION_MAX_SUBQUESTIONS]
