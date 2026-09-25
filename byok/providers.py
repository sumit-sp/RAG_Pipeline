"""Registry mapping a provider name to a callable that builds a fresh client
from a raw API key. Ships with a `groq` entry; register others as needed --
this is what makes the module usable beyond any one provider/project.
"""

from typing import Any, Callable

_REGISTRY: dict[str, Callable[[str], Any]] = {}


def register_provider(name: str, factory: Callable[[str], Any]) -> None:
    """factory(api_key) -> a ready-to-use client instance for that provider."""
    _REGISTRY[name] = factory


def _build_groq_client(api_key: str) -> Any:
    from groq import Groq

    return Groq(api_key=api_key)


register_provider("groq", _build_groq_client)


def build_client(provider: str, api_key: str) -> Any:
    if provider not in _REGISTRY:
        raise ValueError(
            f"No client factory registered for provider {provider!r}. "
            f"Registered providers: {sorted(_REGISTRY)}. "
            f"Call register_provider({provider!r}, your_factory) first."
        )
    return _REGISTRY[provider](api_key)
