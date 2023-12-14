import streamlit as st
from pathlib import Path

from util import get_chroma

st.set_page_config(page_title="Responder / List of Documents")
st.header("Responder - List of Documents")

chroma = get_chroma()

# Your code here

documents = chroma.get()

for i, idx in enumerate(documents['ids']):
    metadata = documents['metadatas'][i]
    label = f"Document: {Path(metadata['source']).name}, Page: {metadata['page']}"
    with st.expander(label):
        st.write(documents['documents'][i])