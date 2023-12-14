import streamlit as st
from pathlib import Path

from util import get_chroma, register_document

st.set_page_config(page_title="Responder / Documents Upload")
st.header("Responder - document Upload")

DOC_STORE = Path("documents").absolute()

# Your code here

with st.form("upload"):
    uploaded_file = st.file_uploader("Upload a document", type="pdf")
    submit_button = st.form_submit_button(label="Upload")

    if uploaded_file is not None:
        DOC_STORE.mkdir(exist_ok=True, parents=True)
        document_path = DOC_STORE / uploaded_file.name

        if document_path.exists():
            st.error("Document already exists!")
            st.stop()

        with st.spinner("Registering document..."):
            chroma = get_chroma()
            with open(document_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            register_document(chroma, document_path)

        st.success(f"Document uploaded {uploaded_file.name}!")