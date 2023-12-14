from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableParallel


from pathlib import Path
import dotenv

dotenv.load_dotenv()

question = "Opisz jak odbywa się trening modeli LLM."
model = ChatOpenAI(temperature=0.0)

# --- Load and index documents ---
file_path = Path(__file__).parent / "blog-entry.pl.pdf"
loader = PyPDFLoader(str(file_path))
chunks = loader.load_and_split()

db = Chroma(embedding_function=OpenAIEmbeddings())
db.add_documents(chunks)
retriever = db.as_retriever()

# --- Prepare prompt ---

template = """Odpowiedź na pytanie używając tylko następującego kontekstu:

{context}

Pytanie: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# --- Utils ---


def format_docs(documents):
    return "\n\n".join([d.page_content for d in documents])


# --- Chain definition ---
query_chain = (
    {
        "context": lambda data: format_docs(data["documents"]),
        "question": lambda data: data["question"],
    }
    | prompt
    | model
    | StrOutputParser()
)


chain = RunnableParallel(
    {"documents": retriever, "question": RunnablePassthrough()}
) | {
    "documents": lambda data: [d.metadata for d in data["documents"]],
    "answer": query_chain,
}

# --- Run chain ---

print("----- Chain with model and documents: -----")
response = chain.invoke(question)
print("Response:", response["answer"])
print("Used documents:")
for doc in response["documents"]:
    print(doc)
