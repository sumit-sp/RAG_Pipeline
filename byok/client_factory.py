"""Resolves a ready-to-use provider client from either a vault session or a
server-side fallback key -- the one function application code actually calls.
"""

from typing import Any

from byok.providers import build_client
from byok.vault import KeyVault


class NoKeyAvailable(Exception):
    """Raised when no session key was found and no fallback_key was given."""


def get_client(
    vault: KeyVault,
    provider: str,
    session_id: str | None = None,
    fallback_key: str | None = None,
) -> Any:
    """Returns a freshly-built client for `provider`, never a cached/shared
    one -- safe to call once per request even when many different sessions'
    keys are in play concurrently, since nothing here is stored or reused
    across calls other than inside the vault itself."""
    key = vault.get_key(session_id, provider=provider) if session_id else None
    if key is None:
        key = fallback_key
    if key is None:
        raise NoKeyAvailable(
            f"No API key available for provider {provider!r} -- no valid "
            f"session_id given (or it expired) and no fallback_key configured."
        )
    return build_client(provider, key)
