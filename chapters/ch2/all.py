from langchain.chat_models import ChatOpenAI
import dotenv

dotenv.load_dotenv()

model = ChatOpenAI(temperature=0.0)

print(model.invoke("Przywitaj się z uczestnikami naszego kursu."))
print(model.invoke("Przywitaj się z uczestnikami naszego kursu."))

from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from pathlib import Path

file_path = Path(__file__).parent / "blog-entry.pl.pdf"
loader = PyPDFLoader(str(file_path))

chunks = loader.load_and_split()

db = Chroma(embedding_function=OpenAIEmbeddings())
chunk_ids = db.add_documents(chunks)

question = "Opisz jak odbywa się trening modeli LLM."

documents = db.similarity_search(question, k=3)
print("Most similar documents:")
for doc in documents:
    print(doc.page_content[:100] + '...')

context = "\n\n".join([d.page_content for d in documents])

prompt = f"""Odpowiedź na pytanie używając tylko następującego kontekstu: 

{context}

Pytanie: {question}
"""

print()
print("----- Prompt: -----")
print(prompt)

answer = model.invoke(prompt)
print()
print("----- Answer: -----")
print(answer)

from langchain.prompts import ChatPromptTemplate

template = """Odpowiedź na pytanie używając tylko następującego kontekstu:

{context}

Pytanie: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
result = prompt.invoke({"context": "Wypełnimy później", "question": question})

print(type(result), result)

retriever = db.as_retriever()
result = retriever.invoke(question)
print(type(result), result)


from langchain.schema.runnable import RunnablePassthrough


def format_docs(documents):
    return "\n\n".join([d.page_content for d in documents])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
)

print()
print("----- Chain: -----")
print(chain.invoke(question))


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
)

print()
print("----- Chain with model: -----")
print(chain.invoke(question))


from langchain.schema import StrOutputParser

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

print()
print("----- Chain with model and parser: -----")
print(chain.invoke(question))

from langchain.schema.runnable import RunnableParallel


query_chain = (
    {
        "context": lambda data: format_docs(data["documents"]),
        "question": lambda data: data["question"]
    }
    | prompt
    | model
    | StrOutputParser()
)


chain = RunnableParallel(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda data: [d.metadata for d in data["documents"]],
    "answer": query_chain
}

print()
print("----- Chain with model and documents: -----")
response = chain.invoke(question)
print("Response:", response["answer"])
print("Used documents:")
for doc in response["documents"]:
    print(doc)