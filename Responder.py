import streamlit as st

from util import get_chroma, get_language_templates, build_chain
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from pathlib import Path

load_dotenv()

st.set_page_config(page_title="Responder", layout="wide")
st.header("Responder - Ask about your documents")

chroma = get_chroma()
retriever = chroma.as_retriever()

# Your code here
with st.sidebar:
    templates = get_language_templates()
    template_key = st.selectbox("Select prompt language", templates.keys())
    template = templates[template_key]

    model_name = st.selectbox("Select model", ["gpt-3.5-turbo", "gpt-4"])
    model = ChatOpenAI(temperature=0.0, model=model_name)


if query := st.chat_input("What is your query?"):
    prompt = ChatPromptTemplate.from_template(template)

    chain = build_chain(retriever, prompt, model)

    col_1, col_2 = st.columns(2)
    with col_1:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                output = chain.invoke(query)
                
            st.write(output["answer"])

    with col_2:
        st.subheader("Used documents")
        for metadata in output["documents"]:
            q = {"$and": [{'source': metadata['source']}, {'page': metadata['page']}]}
            content = chroma.get(where=q)
            label = f"Document: {Path(metadata['source']).name}, Page: {metadata['page']}"
            with st.expander(label):
                st.write(content['documents'][0])