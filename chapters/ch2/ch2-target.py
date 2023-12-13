from langchain.chat_models import ChatOpenAI
import dotenv

dotenv.load_dotenv()

model = ChatOpenAI(temperature=0.0)

# print(model.invoke("Przywitaj się z uczestnikami naszego kursu."))
# print(model.invoke("Przywitaj się z uczestnikami naszego kursu."))

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