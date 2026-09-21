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
