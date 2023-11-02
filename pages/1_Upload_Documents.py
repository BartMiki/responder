import streamlit as st
import tempfile

from util import get_chroma, get_document_store

st.set_page_config(
    page_title="Responder / Documents Upload"
)

st.header("Responder - document Upload")


with st.form("upload"):
    uploaded_file = st.file_uploader("Upload a document", type="pdf")
    document_name = st.text_input("Document name", value="", placeholder="Document name. Leave empty for default file name.")
    submit_button = st.form_submit_button(label="Upload")

    # uploaded_file = st.sidebar.file_uploader("Upload a document", type="pdf")
    if uploaded_file is not None:
        chroma = get_chroma()
        store = get_document_store(chroma)
        document_name = document_name or uploaded_file.name

        if store.exists_by_name(document_name):
            st.error("Document already exists!")
            st.stop()

        else:
            with st.spinner("Registering document..."):
                doc_hash = store.add_document(document_name, uploaded_file.getbuffer())
                
            st.success(f"Document uploaded! Hash: {doc_hash}")