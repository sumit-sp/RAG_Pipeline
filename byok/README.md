# byok

A standalone, framework-agnostic session-key vault for "bring your own API
key" features in AI/GenAI/agentic-AI apps. Takes a user-supplied LLM API key
once, holds it securely for the life of a session, and hands back a
ready-to-use client on demand -- without ever writing the key to disk,
logging it, or letting one session's key leak into another request.

No dependency on this repo beyond `cachetools` and whichever provider SDK(s)
you register (e.g. `groq`, `openai`, `anthropic`) -- copy this directory into
any other project unmodified.

## Quick start

```python
from byok import KeyVault, get_client, NoKeyAvailable

# Construct once per process, share the same instance across all requests.
vault = KeyVault(ttl_seconds=1800)  # 30 minutes

# When a user submits their key (e.g. a "connect your API key" form):
session_id = vault.create_session("groq", user_supplied_key)
# Give session_id back to the caller (e.g. store it in their browser session).
# Never send the raw key back, and never store it anywhere yourself beyond
# the vault -- session_id is the only thing that should travel from here on.

# On every subsequent request, resolve a client from the session:
client = get_client(vault, "groq", session_id, fallback_key=SERVER_DEFAULT_KEY)
client.chat.completions.create(...)  # use exactly like the real SDK client
```

`get_client` builds a **fresh client every call** -- nothing is cached or
reused across requests other than inside the vault itself, so this is safe
to call once per request even with many different sessions' keys in flight
concurrently.

## Adding another provider

```python
from byok import register_provider

register_provider("openai", lambda api_key: OpenAI(api_key=api_key))
```

## Security properties

- The raw key is only ever held **in memory** -- never written to disk or a
  database.
- Sessions **auto-expire** via TTL (`KeyVault(ttl_seconds=...)`); the whole
  vault is wiped on process restart.
- The raw key is transmitted from caller to the vault **exactly once** (at
  `create_session`) -- callers pass `session_id` on every later call, not the
  key itself.
- `session_id` is a high-entropy, unguessable token (`secrets.token_urlsafe`)
  -- the same trust model as a session cookie or capability token. Anyone who
  obtains a live `session_id` can use the key behind it for the remaining TTL
  window; treat it the way you'd treat any other session credential.
- Nothing in this module logs the key. If you wire it into an app with its
  own logging/tracing (structured logs, an APM, an LLM observability tool),
  audit those call sites yourself -- this module can't guarantee what code
  outside it chooses to log.

## Known limitation

`KeyVault` is a single-process, in-memory store -- sessions are **not**
shared across multiple server instances/processes. Fine for a single-instance
deployment; for horizontal scaling, swap the backing store for something
shared (e.g. Redis with a TTL set per key) behind the same `KeyVault`
interface (`create_session`/`get_key`/`revoke`).
