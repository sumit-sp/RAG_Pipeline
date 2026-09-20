"""DeepEval test suite for the plain-backend RAG pipeline.

Run with: deepeval test run eval/run_eval.py
Reads eval/golden_set.jsonl (one question/expected-answer pair per line) and, for
each, actually runs retrieval + generation against the live pipeline, then scores
the result with DeepEval's faithfulness/relevancy/precision/recall metrics.

Phase 2 rule: this file measures the pipeline, it does not improve it. Threshold
tuning and retrieval/generation changes belong in Phase 3.
"""

import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    GEval,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams

from app.core import config
from app.pipelines.plain.generation import PlainGenerator
from app.pipelines.plain.retrieval import PlainRetriever
from eval.judge_model import GroqJudgeModel

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
METRIC_THRESHOLD = 0.5

# assert_test doesn't expose per-case scores when run via plain pytest (that data
# normally lives in DeepEval's own `deepeval test run` reporting layer). Since
# eval/results.md needs actual numbers, not just pass/fail, every run appends
# each case's raw metric scores here so they can be aggregated afterwards.
# Named per retrieval config so concurrent/overlapping runs can't corrupt each
# other (a real bug hit twice during Phase 3 — see DECISIONS.md).
_RUN_LABEL = config.RETRIEVAL_MODE + ("_rerank" if config.USE_RERANKING else "")
RAW_RESULTS_PATH = Path(__file__).parent / f"eval_run_raw_results_{_RUN_LABEL}.jsonl"


def _load_golden_set() -> list[dict]:
    if not GOLDEN_SET_PATH.exists():
        return []
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_golden_set = _load_golden_set()
_judge = GroqJudgeModel()
_retriever = PlainRetriever()
_generator = PlainGenerator()


def _retrieval_hit(item: dict, contexts: list) -> bool:
    """Component-level retriever check (reference-based, programmatic — no LLM
    judge needed): did the expected source document actually come back in top-k?
    Independent of the pipeline-level LLM-judged metrics below, which can look
    fine even when retrieval missed the "right" document, as long as generation
    stayed faithful to whatever it did retrieve instead."""
    retrieved_docs = {c.chunk.source_doc for c in contexts}
    return item["expected_source_doc"] in retrieved_docs


# Application-level: does the actual answer match the golden answer, not just "is
# it faithful to whatever was retrieved" (Faithfulness) or "is it on-topic"
# (Answer Relevancy) — neither of those checks correctness against expected_answer.
_answer_correctness = GEval(
    name="Answer Correctness",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
    ],
    criteria=(
        "Determine whether 'actual output' is factually correct and consistent with "
        "'expected output', given the question in 'input'. The core facts (dates, "
        "obligations, article/section numbers, yes/no answers) must match. Extra "
        "correct detail, different phrasing, or a different level of verbosity are "
        "fine and should not be penalized."
    ),
    threshold=METRIC_THRESHOLD,
    model=_judge,
)


def _append_raw_result(question: str, difficulty: str, metrics: list, retrieval_hit: bool) -> None:
    record = {
        "question": question,
        "difficulty": difficulty,
        "scores": {m.__name__: m.score for m in metrics},
        "retrieval_hit": retrieval_hit,
    }
    with RAW_RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@pytest.mark.parametrize(
    "item", _golden_set, ids=[item["question"][:60] for item in _golden_set]
)
def test_rag_pipeline(item):
    contexts = _retriever.retrieve(item["question"])
    answer = _generator.generate(item["question"], contexts)
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=answer.text,
        expected_output=item["expected_answer"],
        retrieval_context=[c.chunk.text for c in contexts],
    )

    hit = _retrieval_hit(item, contexts)

    metrics = [
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=_judge),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=_judge),
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=_judge),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=_judge),
        _answer_correctness,
    ]
    for metric in metrics:
        metric.measure(test_case)
    _append_raw_result(item["question"], item["difficulty"], metrics, hit)

    assert hit, f"Expected source doc {item['expected_source_doc']!r} was not retrieved (component-level retriever check)"
    assert_test(test_case, metrics)
