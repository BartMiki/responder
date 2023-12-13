from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader

import dotenv

dotenv.load_dotenv()


embedding_function = OpenAIEmbeddings()
documents = [
    "Filemon to biały kot", 
    "Rex to czarny pies"
]

embeddings = embedding_function.embed_documents(documents)

for embedding in embeddings:
    print(f"Size: {len(embedding)}, ten first values: {embedding[:10]}")

doc_to_embedding = dict(zip(documents, embeddings))
question = "Czym jest Filemon?"

question_embedding = embedding_function.embed_query(question)

def cosine_similarity(a, b):
    import numpy as np
    from numpy.linalg import norm

    return np.dot(a, b) / (norm(a) * norm(b))

doc_similarities = {
    doc: cosine_similarity(question_embedding, embedding)
    for doc, embedding in doc_to_embedding.items()
}

print(f"Similarities to question '{question}':")
for doc, similarity in doc_similarities.items():
    print(f"{doc}: {similarity}")
    

# db = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="db")
db = Chroma(embedding_function=OpenAIEmbeddings())
index = db.add_texts(documents)

for idx in index:
    record = db.get(index, include=['documents', 'embeddings'])
    print(f"Index: {idx}, Document: '{record['documents'][0]}', Embedding[:5]: {record['embeddings'][0][:5]}")


doc_similarities = db.similarity_search_with_relevance_scores(question)
print(f"Similarities to question '{question}' using database:")
for doc, similarity in doc_similarities:
    print(f"{doc.page_content}: {similarity}")


from pathlib import Path
file_path = Path(__file__).parent / "blog-entry.pl.pdf"
loader = PyPDFLoader(str(file_path))

chunks = loader.load_and_split()
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(chunk)

chunk_ids = db.add_documents(chunks)
records = db.get(chunk_ids)

print("-"*80)

print("Database content:")
for i, idx in enumerate(records['ids']):
    print(f"Index: {idx}, Document: '{records['documents'][i]}', Metadata: {records['metadatas'][i]}")
    print()
