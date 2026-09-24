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

question = st.text_input("Your question")


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
                response = requests.post(f"{API_URL}/query", json={"question": question}, timeout=120)
                response.raise_for_status()
                data = response.json()
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
