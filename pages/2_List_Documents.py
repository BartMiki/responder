import streamlit as st
import tempfile

from util import get_chroma, get_document_store

st.set_page_config(
    page_title="Responder / List of Documents"
)

st.header("Responder - List of Documents")

chroma = get_chroma()
store = get_document_store(chroma)

for name, file_path in store.list_documents():
    file_name = name
    if not name.endswith(".pdf"):
        file_name += ".pdf"

    with open(file_path, "rb") as f:
        st.download_button(name, f, file_name=file_name, mime="application/pdf")
