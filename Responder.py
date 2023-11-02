import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from util import get_chroma

st.set_page_config(page_title="Responder")

st.header("Responder - Ask about your documents")


chroma = get_chroma()
retriever = chroma.as_retriever()

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


if query := st.chat_input("What is your query?"):
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(temperature=0.0)

    chain = (
        {"context": retriever | format_docs} | {"question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            output = chain.invoke(query)
            
        st.write(output)
