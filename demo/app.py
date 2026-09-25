"""Minimal Streamlit UI hitting the FastAPI /query endpoint. Phase 1: ugly is fine."""

import os
import time

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="EU AI Act Compliance Assistant")
st.title("EU AI Act Compliance Assistant")
st.caption(
    "Ask about obligations under the EU AI Act, the Digital Omnibus amendment, "
    "GPAI guidance, or GDPR. Answers are grounded only in the ingested corpus."
)

_BACKEND_WAKE_BUDGET_SECONDS = 90  # matches _wait_for_backend's own budget below


def _check_backend_once() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=8).status_code == 200
    except requests.RequestException:
        return False


def _render_backend_status() -> None:
    """Pings /health on page load (and automatically again every few seconds
    while not yet ready, via st.rerun()) so a visitor sees whether the
    backend is awake *before* they ask anything -- Render's free tier spins
    it down after inactivity, and the first request after that can take up
    to about a minute to wake. Caches the result in session_state so an
    already-confirmed-ready session doesn't re-ping on every rerun (e.g. from
    typing in another field)."""
    if "backend_wake_deadline" not in st.session_state:
        st.session_state.backend_wake_deadline = time.time() + _BACKEND_WAKE_BUDGET_SECONDS
        st.session_state.backend_status = "checking"

    placeholder = st.empty()

    if st.session_state.backend_status == "ready":
        placeholder.markdown("🟢 **Backend ready**")
        return

    ready = _check_backend_once()
    if ready:
        st.session_state.backend_status = "ready"
        placeholder.markdown("🟢 **Backend ready**")
        return

    if time.time() < st.session_state.backend_wake_deadline:
        st.session_state.backend_status = "waking"
        placeholder.markdown("🟡 **Waking up backend...** (free tier can take up to a minute)")
        time.sleep(3)
        st.rerun()
    else:
        st.session_state.backend_status = "down"
        placeholder.markdown("🔴 **Backend unreachable** — it may still be starting up")
        if st.button("🔄 Check again"):
            st.session_state.backend_wake_deadline = time.time() + _BACKEND_WAKE_BUDGET_SECONDS
            st.rerun()


_render_backend_status()

with st.expander("Optional: use your own Groq API key"):
    st.caption(
        "By default, questions use this demo's shared Groq key. Paste your own "
        "[Groq API key](https://console.groq.com/keys) to use your own quota "
        "instead. Your key is sent to the backend once, to start a session "
        "(see byok/README.md) -- it is never stored on disk, logged, or sent "
        "again after that; only an opaque session id is reused for later questions."
    )
    byok_key = st.text_input("Groq API key", type="password", key="byok_key_input")

question = st.text_input("Your question")


def _ensure_session(byok_key: str) -> str | None:
    """Exchanges a raw key for a session_id at most once per distinct key
    value -- reuses the existing session_id across reruns/questions as long
    as the pasted key hasn't changed. Returns None (use the shared server
    key) if the field is empty."""
    if not byok_key.strip():
        st.session_state.pop("byok_session_id", None)
        st.session_state.pop("byok_key_for_session", None)
        return None

    if st.session_state.get("byok_key_for_session") == byok_key and st.session_state.get(
        "byok_session_id"
    ):
        return st.session_state["byok_session_id"]

    response = requests.post(
        f"{API_URL}/session", json={"provider": "groq", "api_key": byok_key}, timeout=15
    )
    response.raise_for_status()
    session_id = response.json()["session_id"]
    st.session_state["byok_session_id"] = session_id
    st.session_state["byok_key_for_session"] = byok_key
    return session_id


def _wait_for_backend(spinner_status, max_wait_seconds: int = 90) -> bool:
    """Render's free tier spins the API down after inactivity. The first
    request after that can 503 with a hibernate-wake-error header, or the
    API's own lazy pipeline build (loading embedding models, connecting to
    Qdrant) can take a while on a cold instance -- both comfortably exceed a
    typical request timeout. Poll /health first so the real /query call only
    fires once the backend is actually responding."""
    deadline = time.time() + max_wait_seconds
    waited_any = False
    while time.time() < deadline:
        try:
            if requests.get(f"{API_URL}/health", timeout=10).status_code == 200:
                return True
        except requests.RequestException:
            pass
        waited_any = True
        spinner_status.update(label="Waking up the backend (free tier can take up to a minute)...")
        time.sleep(5)
    return not waited_any  # if we never even got here, let the caller try anyway


if st.button("Ask") and question.strip():
    with st.status("Retrieving and generating...", expanded=False) as status:
        backend_ready = _wait_for_backend(status)
        if not backend_ready:
            status.update(label="Backend didn't wake up in time.", state="error")
            st.error("The backend didn't wake up in time. Please try again in a moment.")
        else:
            status.update(label="Retrieving and generating...")
            try:
                session_id = _ensure_session(byok_key)
            except requests.RequestException as e:
                status.update(label="Couldn't start a session with that key.", state="error")
                st.error(f"Couldn't start a session with the API key you provided: {e}")
            else:
                payload = {"question": question}
                if session_id:
                    payload["session_id"] = session_id
                try:
                    response = requests.post(f"{API_URL}/query", json=payload, timeout=120)
                    response.raise_for_status()
                    data = response.json()
                except requests.HTTPError as e:
                    status.update(label="Request failed.", state="error")
                    if e.response is not None and e.response.status_code in (400, 401, 429):
                        # /query's own distinguishable BYOK error messages
                        # (bad/expired session, invalid key, rate limit) --
                        # surface them as-is rather than a generic failure.
                        st.error(e.response.json().get("detail", str(e)))
                    else:
                        st.error(f"Request to API failed: {e}")
                except requests.RequestException as e:
                    status.update(label="Request failed.", state="error")
                    st.error(f"Request to API failed: {e}")
                else:
                    status.update(label="Done.", state="complete")
                    st.markdown("### Answer")
                    st.write(data["answer"])
                    if data["sources"]:
                        st.markdown("### Sources")
                        for source in data["sources"]:
                            st.write(f"- {source}")
