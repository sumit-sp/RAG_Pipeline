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
)
from deepeval.test_case import LLMTestCase

from app.pipelines.plain.generation import PlainGenerator
from app.pipelines.plain.retrieval import PlainRetriever
from eval.judge_model import GroqJudgeModel

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.jsonl"
METRIC_THRESHOLD = 0.5


def _load_golden_set() -> list[dict]:
    if not GOLDEN_SET_PATH.exists():
        return []
    with GOLDEN_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_golden_set = _load_golden_set()
_judge = GroqJudgeModel()
_retriever = PlainRetriever()
_generator = PlainGenerator()


def _run_pipeline(item: dict) -> LLMTestCase:
    contexts = _retriever.retrieve(item["question"])
    answer = _generator.generate(item["question"], contexts)
    return LLMTestCase(
        input=item["question"],
        actual_output=answer.text,
        expected_output=item["expected_answer"],
        retrieval_context=[c.chunk.text for c in contexts],
    )


@pytest.mark.parametrize(
    "item", _golden_set, ids=[item["question"][:60] for item in _golden_set]
)
def test_rag_pipeline(item):
    test_case = _run_pipeline(item)
    assert_test(
        test_case,
        [
            FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=_judge),
            AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=_judge),
            ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=_judge),
            ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=_judge),
        ],
    )
