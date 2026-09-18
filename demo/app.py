"""Minimal Streamlit UI hitting the FastAPI /query endpoint. Phase 1: ugly is fine."""

import os

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

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating..."):
        try:
            response = requests.post(f"{API_URL}/query", json={"question": question}, timeout=60)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            st.error(f"Request to API failed: {e}")
        else:
            st.markdown("### Answer")
            st.write(data["answer"])
            if data["sources"]:
                st.markdown("### Sources")
                for source in data["sources"]:
                    st.write(f"- {source}")
