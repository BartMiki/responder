# Responder
A system for answering user questions based on the internal document base 

## Todo

- [ ] Uploading documents:
  - [ ] Setup a vector database
  - [ ] Setup a file storage
  - [ ] Create an upload form
  - [ ] Calculate doc2vec vector
- [ ] List all documents
- [ ] Accept a user query:
  - [ ] Setup a prompt template
  - [ ] Setup a LLM model (GPT, Lama or Falcon)
  - [ ] Get similar documents from the storage
  - [ ] Enrich prompt with documents summaries and question
  - [ ] Respond
