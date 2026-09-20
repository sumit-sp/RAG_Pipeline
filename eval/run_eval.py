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


def _run_pipeline(item: dict) -> LLMTestCase:
    contexts = _retriever.retrieve(item["question"])
    answer = _generator.generate(item["question"], contexts)
    return LLMTestCase(
        input=item["question"],
        actual_output=answer.text,
        expected_output=item["expected_answer"],
        retrieval_context=[c.chunk.text for c in contexts],
    )


def _append_raw_result(question: str, difficulty: str, metrics: list) -> None:
    record = {
        "question": question,
        "difficulty": difficulty,
        "scores": {m.__name__: m.score for m in metrics},
    }
    with RAW_RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


@pytest.mark.parametrize(
    "item", _golden_set, ids=[item["question"][:60] for item in _golden_set]
)
def test_rag_pipeline(item):
    test_case = _run_pipeline(item)
    metrics = [
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=_judge),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=_judge),
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=_judge),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=_judge),
    ]
    for metric in metrics:
        metric.measure(test_case)
    _append_raw_result(item["question"], item["difficulty"], metrics)
    assert_test(test_case, metrics)
