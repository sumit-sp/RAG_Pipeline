"""byok -- a standalone, framework-agnostic session-scoped API key vault.

Lets end users "bring their own" LLM API key for a session instead of
sharing your server's key, without ever persisting the raw key to disk,
logging it, or letting one session's key leak into another request.

Quick start:

    from byok import KeyVault, get_client, NoKeyAvailable

    vault = KeyVault(ttl_seconds=1800)          # construct once, reuse everywhere

    session_id = vault.create_session("groq", user_supplied_key)  # once, at sign-in

    client = get_client(vault, "groq", session_id, fallback_key=SERVER_DEFAULT_KEY)
    client.chat.completions.create(...)          # use exactly like the real SDK client

See README.md for the full design rationale, security properties, and the
single-process/in-memory limitation.
"""

from byok.client_factory import NoKeyAvailable, get_client
from byok.providers import build_client, register_provider
from byok.vault import KeyVault

__all__ = [
    "KeyVault",
    "get_client",
    "register_provider",
    "build_client",
    "NoKeyAvailable",
]
