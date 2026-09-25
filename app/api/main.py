"""FastAPI app wrapping the retrieval + generation pipeline.

PIPELINE_BACKEND selects which pipeline implementation answers queries. Only
"plain" exists until Phase 6 adds "langchain" against the same interfaces.
"""

import logging
import threading
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core import config
from app.core.interfaces import Generator, Retriever
from app.core.tracing import trace_span

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("app.api")

app = FastAPI(title="EU AI Act Compliance Assistant")

_pipeline: tuple[Retriever, Generator] | None = None
_pipeline_lock = threading.Lock()


def _build_pipeline() -> tuple[Retriever, Generator]:
    if config.PIPELINE_BACKEND == "plain":
        from app.pipelines.plain.generation import PlainGenerator
        from app.pipelines.plain.retrieval import PlainRetriever

        return PlainRetriever(), PlainGenerator()
    raise NotImplementedError(
        f"PIPELINE_BACKEND={config.PIPELINE_BACKEND!r} is not implemented yet."
    )


def _get_pipeline() -> tuple[Retriever, Generator]:
    # Built on first request, not at import/startup time. Loading the
    # embedding models + connecting to Qdrant here is slow enough that it
    # blew past Render's own port-open deploy check when done eagerly —
    # ASGI lifespan startup doesn't dodge this either, since uvicorn's port
    # doesn't become connectable until lifespan startup itself completes
    # (verified directly, not assumed). Deferring to first request means the
    # port opens immediately; only the first /query pays the load cost.
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = _build_pipeline()
    return _pipeline


class QueryRequest(BaseModel):
    question: str
    top_k: int = config.RETRIEVAL_TOP_K


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pipeline_backend": config.PIPELINE_BACKEND}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    t0 = time.perf_counter()
    logger.info("query started question=%r top_k=%d", request.question, request.top_k)

    try:
        # One top-level span per request -- everything retrieve()/generate()
        # trace underneath (retrieval, decomposition, generation) nests under
        # this automatically (Langfuse's OTel context propagation), so a
        # single connected trace shows per-component latency as a waterfall
        # instead of disconnected generation events. Any exception raised
        # inside is recorded on the span (status=ERROR) before propagating.
        with trace_span(
            "query", as_type="span", input={"question": request.question, "top_k": request.top_k}
        ) as request_trace:
            retriever, generator = _get_pipeline()
            contexts = retriever.retrieve(request.question, top_k=request.top_k)

            if not contexts:
                response = QueryResponse(
                    answer="No indexed documents found to answer this question yet.",
                    sources=[],
                )
            else:
                answer = generator.generate(request.question, contexts)
                response = QueryResponse(answer=answer.text, sources=answer.citations)

            request_trace.set_output(
                {"answer": response.answer, "sources": response.sources},
                metadata={"chunk_count": len(contexts)},
            )
    except Exception:
        elapsed = time.perf_counter() - t0
        logger.exception(
            "query failed question=%r elapsed_s=%.2f", request.question, elapsed
        )
        raise HTTPException(status_code=500, detail="Internal error answering this question.")

    elapsed = time.perf_counter() - t0
    logger.info(
        "query completed question=%r elapsed_s=%.2f chunk_count=%d",
        request.question, elapsed, len(contexts),
    )
    return response
