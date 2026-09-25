"""Langfuse tracing for every generation call (Langfuse Cloud, not self-hosted —
see DECISIONS.md). No-ops entirely if LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY
aren't set, so local dev without a Langfuse account works unchanged.
"""

from contextlib import contextmanager
from typing import Any

from app.core import config

_client = None
if config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY:
    from langfuse import Langfuse

    _client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
    )


class _Generation:
    def __init__(self, span: Any | None):
        self._span = span

    def set_output(self, text: str, usage: dict[str, int] | None = None) -> None:
        if self._span is None:
            return
        kwargs: dict[str, Any] = {"output": text}
        if usage:
            kwargs["usage_details"] = usage
        self._span.update(**kwargs)


@contextmanager
def trace_generation(
    name: str,
    input: Any,
    model: str,
    model_parameters: dict[str, Any] | None = None,
):
    """Wraps one LLM call. Yields a `_Generation` with `.set_output(text, usage)`."""
    if _client is None:
        yield _Generation(None)
        return

    with _client.start_as_current_observation(
        name=name,
        as_type="generation",
        input=input,
        model=model,
        model_parameters=model_parameters,
    ) as span:
        yield _Generation(span)


class _Span:
    def __init__(self, span: Any | None):
        self._span = span

    def set_output(self, output: Any, metadata: dict[str, Any] | None = None) -> None:
        if self._span is None:
            return
        kwargs: dict[str, Any] = {"output": output}
        if metadata:
            kwargs["metadata"] = metadata
        self._span.update(**kwargs)


@contextmanager
def trace_span(
    name: str,
    as_type: str = "span",
    input: Any = None,
    metadata: dict[str, Any] | None = None,
):
    """Wraps one non-generation pipeline step (retrieval, a retrieval
    sub-component, decomposition's retrieval fan-out, the whole /query
    request). Yields a `_Span` with `.set_output(output, metadata)`.

    Nests automatically under whatever span is currently open (Langfuse uses
    OpenTelemetry context propagation), so wrapping the top-level /query
    handler in one of these and calling retrieve()/generate() underneath
    produces one connected trace per request instead of disconnected
    generation events -- each nested span's own start/end gives per-component
    latency for free, visible as a waterfall in Langfuse's trace view.

    An exception raised inside the `with` block is recorded on the span
    (status=ERROR) by the underlying OTel context manager and re-raised
    unchanged -- this file doesn't swallow errors, so tracing failures
    doesn't mean losing the actual exception."""
    if _client is None:
        yield _Span(None)
        return

    with _client.start_as_current_observation(
        name=name,
        as_type=as_type,
        input=input,
        metadata=metadata,
    ) as span:
        yield _Span(span)
