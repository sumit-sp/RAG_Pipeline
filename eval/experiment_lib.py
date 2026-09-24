"""Reusable helpers behind notebooks/rag_experiments.ipynb (and available to
any future eval/*.py script that wants the same building blocks). Pulls
together retrieval, plain vs. chain-of-thought generation
(eval/cot_generation.py), and query decomposition into one `run_variant`
call so the notebook doesn't have to re-derive corpus/config wiring or
span-based scoring (eval/span_scoring.py) inline.

Nothing here calls an LLM to judge or score -- Precision/Recall is the
deterministic span-overlap check from eval/span_scoring.py, and `hit` is a
plain set-membership check against golden_set.jsonl's expected_source_doc.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from app.core import config
from app.core.models import Answer, RetrievedContext
from app.pipelines.plain.generation import PlainGenerator
from app.pipelines.plain.retrieval import PlainRetriever
from eval.cot_generation import CotGenerator
from eval.span_scoring import ChunkSpanLookup, load_qrels_spans, precision_recall

EVAL_DIR = Path(__file__).parent
GOLDEN_SET_PATH = EVAL_DIR / "golden_set.jsonl"
DECOMPOSED_PATH = EVAL_DIR / "decomposed_questions.jsonl"

# name -> (QDRANT_COLLECTION base, CHUNKING_STRATEGY, USE_CONTEXTUAL_HEADERS, CONTEXTUAL_HEADERS_PATH)
# Mirrors the named configs documented in DEPLOYMENT.md / EVALUATION_HISTORY.md.
_CORPORA: dict[str, tuple[str, str, bool, str | None]] = {
    "ai_act_corpus": ("ai_act_corpus", "fixed", True, "eval/contextual_headers.jsonl"),
    "ai_act_corpus_fixed": ("ai_act_corpus_fixed", "fixed", False, None),
    "ai_act_corpus_recursive": ("ai_act_corpus_recursive", "recursive", False, None),
    "ai_act_corpus_recursive_ctxheaders": (
        "ai_act_corpus_recursive_ctxheaders",
        "recursive",
        True,
        "eval/contextual_headers_recursive.jsonl",
    ),
}

_lookup_cache: dict[str, ChunkSpanLookup] = {}


LOCAL_QDRANT_PATH = str(EVAL_DIR / "local_qdrant_data")


def use_local_qdrant(path: str = LOCAL_QDRANT_PATH) -> None:
    """Forces app.core.vector_store.get_qdrant_client() to use qdrant-client's
    embedded/on-disk mode instead of the real Qdrant Cloud cluster -- no
    network required. Call this BEFORE set_corpus()/PlainRetriever()/
    PlainIngestor(), since the client is built once in their __init__.

    Existing-`Qdrant_URL`-based collections and this local, on-disk path are
    entirely separate storage -- nothing here reads or writes the Cloud
    cluster. Use eval/build_local_corpus.py once to populate `path` with a
    production-config-matching corpus before running experiments against it."""
    config.QDRANT_URL = None
    config.QDRANT_PATH = path


def list_corpora() -> None:
    print("Named corpora available to set_corpus():")
    for name, (collection, strategy, headers, headers_path) in _CORPORA.items():
        headers_note = f", headers={headers_path}" if headers else ""
        print(f"  {name!r:45s} -> collection={collection}, chunking={strategy}{headers_note}")


def set_corpus(name: str, local: bool = False) -> PlainRetriever:
    """Mutates the global app.core.config module (same pattern as
    eval/query_decomposition_experiment.py) and returns a fresh PlainRetriever
    pointed at it. Call this again after changing corpus/chunking strategy --
    a retriever built before the switch is still wired to the old collection.

    `local=True` points at the embedded, on-disk Qdrant copy built by
    eval/build_local_corpus.py (eval/local_qdrant_data/) instead of Qdrant
    Cloud -- use this while Cloud connectivity is down. Only
    "ai_act_corpus_recursive_ctxheaders" has been built locally so far; other
    names will raise until eval/build_local_corpus.py is extended or re-run
    for them."""
    if name not in _CORPORA:
        raise ValueError(f"Unknown corpus {name!r}. Options: {list(_CORPORA)}")
    if local:
        use_local_qdrant()
    collection, strategy, use_headers, headers_path = _CORPORA[name]
    config.QDRANT_COLLECTION = collection
    config.CHUNKING_STRATEGY = strategy
    config.USE_CONTEXTUAL_HEADERS = use_headers
    if headers_path:
        config.CONTEXTUAL_HEADERS_PATH = headers_path
    return PlainRetriever()


def load_golden_set() -> dict[str, dict]:
    return {
        item["question"]: item
        for item in (
            json.loads(line)
            for line in GOLDEN_SET_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def load_decomposed_questions() -> dict[str, list[str]]:
    if not DECOMPOSED_PATH.exists():
        return {}
    return {
        item["question"]: item["sub_questions"]
        for item in (
            json.loads(line)
            for line in DECOMPOSED_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _merge_dedupe(context_lists: list[list[RetrievedContext]]) -> list[RetrievedContext]:
    seen_ids = set()
    merged = []
    for contexts in context_lists:
        for c in contexts:
            if c.chunk.id not in seen_ids:
                merged.append(c)
                seen_ids.add(c.chunk.id)
    return merged


def _span_lookup() -> ChunkSpanLookup:
    key = f"{config.CHUNKING_STRATEGY}:{config.CHUNK_SIZE_TOKENS}:{config.CHUNK_OVERLAP_TOKENS}"
    if key not in _lookup_cache:
        _lookup_cache[key] = ChunkSpanLookup()
    return _lookup_cache[key]


@dataclass
class VariantRun:
    variant_label: str
    question: str
    n_chunks: int
    hit: bool | None  # None if no golden_set entry was supplied
    precision_recall: tuple[float, float] | None
    answer: Answer
    contexts: list[RetrievedContext]


def run_variant(
    retriever: PlainRetriever,
    question: str,
    sub_questions: list[str] | None = None,
    use_cot: bool = False,
    top_k: int = config.RETRIEVAL_TOP_K,
    variant_label: str | None = None,
    golden_set: dict[str, dict] | None = None,
) -> VariantRun:
    """One retrieve+generate run. `sub_questions=None` retrieves once for
    `question` as-is; passing a list retrieves once per sub-question and
    merges+dedupes (Step 19's decomposition pattern)."""
    queries = sub_questions if sub_questions else [question]
    contexts = _merge_dedupe([retriever.retrieve(q, top_k=top_k) for q in queries])

    generator = CotGenerator() if use_cot else PlainGenerator()
    answer = generator.generate(question, contexts)

    qrels = load_qrels_spans().get(question, [])
    pr = precision_recall(contexts, qrels, _span_lookup()) if qrels else None

    hit = None
    if golden_set and question in golden_set:
        expected = golden_set[question]["expected_source_doc"]
        hit = expected in {c.chunk.source_doc for c in contexts}

    label = variant_label or ("decomposed" if sub_questions else "baseline") + ("_cot" if use_cot else "")
    return VariantRun(
        variant_label=label,
        question=question,
        n_chunks=len(contexts),
        hit=hit,
        precision_recall=pr,
        answer=answer,
        contexts=contexts,
    )


def summarize_runs(runs: list[VariantRun]):
    """Returns a pandas DataFrame -- one row per VariantRun -- for display in
    a notebook cell. Imports pandas lazily so eval/*.py scripts that use this
    module for run_variant() alone don't need it installed."""
    import pandas as pd

    rows = [
        {
            "question": r.question[:60] + ("..." if len(r.question) > 60 else ""),
            "variant": r.variant_label,
            "n_chunks": r.n_chunks,
            "hit": r.hit,
            "precision": r.precision_recall[0] if r.precision_recall else None,
            "recall": r.precision_recall[1] if r.precision_recall else None,
            "citations": ", ".join(r.answer.citations),
            "answer_preview": r.answer.text[:150],
        }
        for r in runs
    ]
    return pd.DataFrame(rows)
