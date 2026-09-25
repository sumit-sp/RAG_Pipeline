"""In-memory, TTL-bound session store for user-supplied API keys.

Design intent: a caller's raw key should touch the server exactly once (at
session creation), live only in memory, auto-expire, and never be logged or
persisted -- see README.md for the full security rationale and the
single-process limitation.
"""

import secrets
import threading
from dataclasses import dataclass

from cachetools import TTLCache

DEFAULT_TTL_SECONDS = 1800  # 30 minutes


@dataclass(frozen=True)
class _Entry:
    provider: str
    api_key: str


class KeyVault:
    """Thread-safe. One vault instance is meant to be a process-wide
    singleton -- construct it once and share it across all requests.

    All sessions in a given vault share the same TTL (a `cachetools.TTLCache`
    limitation, not a design choice) -- construct a separate vault if you
    need a different expiry window for a different use case."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, maxsize: int = 10_000):
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._store: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl_seconds)

    def create_session(self, provider: str, api_key: str) -> str:
        if not api_key or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        session_id = secrets.token_urlsafe(32)
        with self._lock:
            self._store[session_id] = _Entry(provider=provider, api_key=api_key)
        return session_id

    def get_key(self, session_id: str, provider: str | None = None) -> str | None:
        """Returns the raw key for a live session, or None if the session
        doesn't exist, has expired, or (when `provider` is given) doesn't
        match the provider it was created for."""
        with self._lock:
            entry = self._store.get(session_id)
        if entry is None:
            return None
        if provider is not None and entry.provider != provider:
            return None
        return entry.api_key

    def revoke(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
