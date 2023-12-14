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
