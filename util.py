import hashlib
import json
import os
from pathlib import Path
import streamlit as st
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader


@st.cache_resource()
def get_chroma():
    return Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="db")


@st.cache_resource()
def get_document_store(_chroma):
    return DocumentStore("documents", _chroma)


class DocumentStore:
    def __init__(self, directory: str, chroma: Chroma):
        self.directory = Path(directory)
        self.chroma = chroma
        self.directory.mkdir(exist_ok=True, parents=True)

    def _name_to_hash(self, name: str) -> str:
        return hashlib.md5(name.encode()).hexdigest()

    def _get_doc_dir(self, doc_hash: str) -> Path:
        return self.directory / doc_hash

    def exists_by_name(self, name: str) -> bool:
        meta_path = self._get_doc_dir(self._name_to_hash(name)) / "meta.json"
        return meta_path.exists()        

    def add_document(self, name: str, doc: memoryview) -> str:
        doc_hash = self._name_to_hash(name)
        doc_dir = self._get_doc_dir(doc_hash)
        doc_path = doc_dir / "document.pdf"
        meta_path = doc_dir / "meta.json"

        if not meta_path.exists():
            doc_dir.mkdir(exist_ok=True, parents=True)
            with open(doc_path, "wb") as f:
                f.write(doc)

            loader = PyPDFLoader(str(doc_path))
            pages = loader.load_and_split()

            chunk_ids = self.chroma.add_documents(pages)

            with open(meta_path, "w") as f:
                json.dump({"name": name, "chunk_ids": chunk_ids}, f)

        return doc_hash

    def list_documents(self):
        for doc_dir in self.directory.iterdir():
            meta_path = doc_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)

                yield meta["name"], str(doc_dir / "document.pdf")