"""Local experimentation UI -- NOT the deployed demo (see demo/app.py for that).

Runs the pipeline in-process (no FastAPI hop) so it can show retrieval
internals the production API doesn't expose: per-chunk scores, which
retrieval path found each chunk, the exact prompt sent to the model, token
usage, and timing. Meant for iterating on config locally, not for anyone
else to use.

Run with: streamlit run dev_ui/experiment_app.py
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))  # repo root, so `app.*` imports resolve

from app.core import config  # noqa: E402
from app.pipelines.plain.generation import PlainGenerator  # noqa: E402
from app.pipelines.plain.retrieval import PlainRetriever  # noqa: E402
from dev_ui import eval_lookup  # noqa: E402
from dev_ui.pipeline_runner import RunResult, retriever_cache_key, run_pipeline  # noqa: E402

FEEDBACK_PATH = Path(__file__).parent / "relevance_feedback.jsonl"

st.set_page_config(page_title="RAG Experimentation UI", layout="wide")


@st.cache_resource(show_spinner="Loading embedding models / connecting to Qdrant...")
def _get_retriever(cache_key: tuple) -> PlainRetriever:
    return PlainRetriever()


@st.cache_resource(show_spinner="Connecting to Groq...")
def _get_generator() -> PlainGenerator:
    return PlainGenerator()


@st.cache_data
def _golden_set() -> dict:
    return eval_lookup.load_golden_set()


@st.cache_data
def _qrels() -> dict:
    return eval_lookup.load_qrels()


def _log_feedback(question: str, source_doc: str, chunk_index: int, relevant: bool) -> None:
    record = {
        "question": question,
        "source_doc": source_doc,
        "chunk_index": chunk_index,
        "relevant": relevant,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


st.title("RAG Experimentation UI")
st.caption(
    "Local-only, for iterating on retrieval/generation config. Not the deployed demo "
    "(see demo/app.py) and not wired to CI."
)

# ---------------------------------------------------------------------------
# Sidebar: config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Data source")
    qdrant_choice = st.radio(
        "Qdrant target",
        ["Cloud (from .env)", "Local embedded"],
        index=0 if config.QDRANT_URL else 1,
        help="Switching this reloads the retriever -- takes a few seconds.",
    )
    if qdrant_choice == "Local embedded":
        config.QDRANT_URL = None
        config.QDRANT_API_KEY = None
    else:
        config.QDRANT_URL = st.session_state.get("_orig_qdrant_url", config.QDRANT_URL)
        config.QDRANT_API_KEY = st.session_state.get("_orig_qdrant_api_key", config.QDRANT_API_KEY)
    st.session_state.setdefault("_orig_qdrant_url", config.QDRANT_URL)
    st.session_state.setdefault("_orig_qdrant_api_key", config.QDRANT_API_KEY)

    st.divider()
    st.header("Retrieval (live)")
    config.RETRIEVAL_MODE = st.selectbox(
        "Retrieval mode", ["hybrid", "dense"],
        index=["hybrid", "dense"].index(config.RETRIEVAL_MODE),
        help="Requires that mode's collection to already exist (each mode has its own).",
    )
    top_k = st.number_input("top_k", min_value=1, max_value=20, value=config.RETRIEVAL_TOP_K)
    config.USE_CROSS_REFERENCE_BOOST = st.checkbox(
        "Cross-reference boost", value=config.USE_CROSS_REFERENCE_BOOST
    )
    config.CROSS_REFERENCE_BOOST_LIMIT = st.number_input(
        "Boost limit", min_value=1, max_value=20, value=config.CROSS_REFERENCE_BOOST_LIMIT,
        disabled=not config.USE_CROSS_REFERENCE_BOOST,
    )
    config.USE_RERANKING = st.checkbox(
        "Reranking", value=config.USE_RERANKING,
        help="Cross-encoder reranker -- tried and reverted as the pipeline default (see DECISIONS.md), but available here to re-test.",
    )

    st.divider()
    st.header("Generation (live)")
    model_options = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    current_model = config.GENERATION_MODEL
    config.GENERATION_MODEL = st.selectbox(
        "Generation model",
        model_options if current_model in model_options else model_options + [current_model],
        index=(model_options if current_model in model_options else model_options + [current_model]).index(current_model),
    )
    custom_model = st.text_input("...or type a different model id", value="")
    if custom_model.strip():
        config.GENERATION_MODEL = custom_model.strip()

    st.divider()
    st.header("Baked into current collection")
    st.caption("Read-only -- these were fixed at ingestion time. Changing them here would do "
               "nothing without re-running ingestion into a matching collection.")
    st.text(f"Embedding: {config.EMBEDDING_BACKEND} / {config.LOCAL_EMBEDDING_MODEL}")
    st.text(f"Chunking: {config.CHUNKING_STRATEGY} "
            f"({config.CHUNK_SIZE_TOKENS}/{config.CHUNK_OVERLAP_TOKENS} tokens)")
    st.text(f"Contextual headers: {config.USE_CONTEXTUAL_HEADERS} (current env value -- "
            f"may not match what an existing collection was actually built with)")

    cache_key = retriever_cache_key()
    try:
        retriever = _get_retriever(cache_key)
        generator = _get_generator()
        collection = config.collection_name()
        info = retriever.client.get_collection(collection)
        st.success(f"Connected -- '{collection}': {info.points_count} points")
    except Exception as e:
        st.error(f"Could not connect / collection not found: {e}")
        st.stop()

# ---------------------------------------------------------------------------
# Main: question / answer
# ---------------------------------------------------------------------------
question = st.text_input("Question", value=st.session_state.get("_last_question", ""))
run_clicked = st.button("Run", type="primary")

golden_match = _golden_set().get(question)
if golden_match:
    with st.expander("📋 Matches a golden-set question", expanded=False):
        st.write(f"**Expected source doc:** `{golden_match['expected_source_doc']}`")
        st.write(f"**Expected answer:** {golden_match['expected_answer']}")
        st.write(f"**Difficulty:** {golden_match.get('difficulty', 'n/a')}")

if run_clicked and question.strip():
    st.session_state["_last_question"] = question
    with st.spinner("Retrieving and generating..."):
        result: RunResult = run_pipeline(retriever, generator, question, top_k=int(top_k))
    st.session_state["_last_result"] = result

result: RunResult | None = st.session_state.get("_last_result")

if result and result.question == question:
    baseline: RunResult | None = st.session_state.get("_baseline")
    columns = st.columns(2) if baseline else [st.container()]

    def _render(col, r: RunResult, label: str):
        with col:
            if label:
                st.markdown(f"#### {label}")
                st.caption(
                    f"mode={r.retrieval_mode} top_k={r.top_k} boost={r.used_boost} "
                    f"rerank={r.used_reranking} model={r.generation_model}"
                )
            st.markdown("### Answer")
            st.write(r.answer_text)
            if r.citations:
                st.markdown("**Sources:** " + ", ".join(f"`{c}`" for c in r.citations))

            m1, m2, m3 = st.columns(3)
            m1.metric("Retrieval time", f"{r.retrieval_seconds:.2f}s")
            m2.metric("Generation time", f"{r.generation_seconds:.2f}s")
            if r.usage:
                m3.metric("Tokens (in/out)", f"{r.usage['input']}/{r.usage['output']}")

            qrels = _qrels().get(r.question)
            if qrels is not None:
                retrieved = {(c.context.chunk.source_doc, c.context.chunk.chunk_index) for c in r.chunks}
                pr = eval_lookup.precision_recall_f1(retrieved, qrels)
                if pr:
                    p, rec, f1 = pr
                    st.caption(f"Against qrels.jsonl -- Precision: {p:.2f}  Recall: {rec:.2f}  F1: {f1:.2f}")

            st.markdown("### Retrieved chunks")
            for i, sc in enumerate(r.chunks):
                c = sc.context.chunk
                tags = f"`{sc.origin}`"
                if c.references:
                    tags += " " + " ".join(f"`references: {ref}`" for ref in c.references)
                header = f"[{i + 1}] {c.source_doc} (chunk {c.chunk_index}) -- score {sc.context.score:.4f}"
                with st.expander(header):
                    st.markdown(tags)
                    score_bits = []
                    if sc.dense_score is not None:
                        score_bits.append(f"dense={sc.dense_score:.4f}")
                    if sc.sparse_score is not None:
                        score_bits.append(f"sparse={sc.sparse_score:.4f}")
                    if score_bits:
                        st.caption("Component scores: " + ", ".join(score_bits))
                    st.text(c.text)
                    fb1, fb2, _ = st.columns([1, 1, 4])
                    key_base = f"{r.question}-{c.source_doc}-{c.chunk_index}-{label}"
                    if fb1.button("👍 Relevant", key=f"up-{key_base}"):
                        _log_feedback(r.question, c.source_doc, c.chunk_index, True)
                        st.toast("Logged as relevant")
                    if fb2.button("👎 Not relevant", key=f"down-{key_base}"):
                        _log_feedback(r.question, c.source_doc, c.chunk_index, False)
                        st.toast("Logged as not relevant")

            with st.expander("Prompt sent to the model"):
                st.text_area("System", r.prompt_system, height=80, key=f"sys-{label}", disabled=True)
                st.text_area("User", r.prompt_user, height=300, key=f"user-{label}", disabled=True)

    if baseline:
        _render(columns[0], baseline, "Pinned baseline")
        _render(columns[1], result, "Current")
    else:
        _render(columns[0], result, "")

    st.divider()
    bcol1, bcol2 = st.columns(2)
    if bcol1.button("📌 Pin this result as baseline"):
        st.session_state["_baseline"] = result
        st.rerun()
    if baseline and bcol2.button("Clear pinned baseline"):
        del st.session_state["_baseline"]
        st.rerun()
