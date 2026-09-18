"""FastAPI app wrapping the retrieval + generation pipeline.

PIPELINE_BACKEND selects which pipeline implementation answers queries. Only
"plain" exists until Phase 6 adds "langchain" against the same interfaces.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core import config
from app.core.interfaces import Generator, Retriever

app = FastAPI(title="EU AI Act Compliance Assistant")


def _build_pipeline() -> tuple[Retriever, Generator]:
    if config.PIPELINE_BACKEND == "plain":
        from app.pipelines.plain.generation import PlainGenerator
        from app.pipelines.plain.retrieval import PlainRetriever

        return PlainRetriever(), PlainGenerator()
    raise NotImplementedError(
        f"PIPELINE_BACKEND={config.PIPELINE_BACKEND!r} is not implemented yet."
    )


_retriever, _generator = _build_pipeline()


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

    contexts = _retriever.retrieve(request.question, top_k=request.top_k)
    if not contexts:
        return QueryResponse(
            answer="No indexed documents found to answer this question yet.",
            sources=[],
        )

    answer = _generator.generate(request.question, contexts)
    return QueryResponse(answer=answer.text, sources=answer.citations)
