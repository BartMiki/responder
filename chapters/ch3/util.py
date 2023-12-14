import hashlib
import json
import os
from pathlib import Path
import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableParallel

PL_TEMPLATE = """Odpowiedź na pytanie używając tylko następującego kontekstu:

{context}

Pytanie: {question}
"""

EN_TEMPLATE = """Answer the question using only the following context:

{context}

Question: {question}
"""


@st.cache_resource()
def get_chroma():
    return Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="db")


def register_document(db: Chroma, path: Path):
    loader = PyPDFLoader(str(path))
    chunks = loader.load_and_split()
    db.add_documents(chunks)


@st.cache_data()
def get_language_templates():
    return {"Polish": PL_TEMPLATE, "English": EN_TEMPLATE}


def format_docs(documents):
    return "\n\n".join([d.page_content for d in documents])


def build_chain(retriever, prompt, model):
    query_chain = (
        {
            "context": lambda data: format_docs(data["documents"]),
            "question": lambda data: data["question"],
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        "documents": lambda data: [d.metadata for d in data["documents"]],
        "answer": query_chain,
    }
