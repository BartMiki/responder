# Etap 3

Witam na warsztatach poświęconych Generative AI!
W tym etapie dowiemy się jak połączyć przygotowaną przez nas wiedzę z interfejsem graficznym.

## Streamlit

Streamlit to biblioteka, która pozwala na tworzenie interfejsów graficznych w Pythonie.
Pozwala na szybkie prototype'owanie i testowanie pomysłów.

Mamy już gotowy plik `utils.py` który zawiera pomocne funkcje które użyjemy.

Żeby uruchomić naszą aplikację, użyjemy komendy:

```bash
streamlit run workshop.py
```

musimy znajdować się w katalogu `chapters/ch3`. Musimy mieć tutaj także plik `.env` z kluczem API OpenAI.

## Dodawanie dokumentów

Dodajmy dokumenty do bazy danych. Tym razem nasza baza danych będzie przechowywana na dysku, także między restartami aplikacji
dane będą dostępne. Dodawanie dokumentów będzie na osobnej stronie, dlatego plik znajduje się w katalogu:`pages`.

```python
with st.form("upload"):
    uploaded_file = st.file_uploader("Upload a document", type="pdf")
    submit_button = st.form_submit_button(label="Upload")

    if uploaded_file is not None:
        DOC_STORE.mkdir(exist_ok=True, parents=True)
        document_path = DOC_STORE / uploaded_file.name

        if document_path.exists():
            st.error("Document already exists!")
            st.stop()

        with st.spinner("Registering document..."):
            chroma = get_chroma()
            with open(document_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            register_document(chroma, document_path)

        st.success(f"Document uploaded {uploaded_file.name}!")
```

Tworzymy formularz który pozwala przysłać plik PDF. Następnie zapisujemy go lokalnie na dysku
i rejestrujemy w bazie danych. Dodawanie dokumentów odbywa się podobnie jak w poprzednim etapie.

## Wyświetlanie dokumentów

Zróbmy dodatkową stronę która pozwoli na wyświetlanie zapisanych dokumentów.

```python
documents = chroma.get()

for i, idx in enumerate(documents['ids']):
    metadata = documents['metadatas'][i]
    label = f"Document: {Path(metadata['source']).name}, Page: {metadata['page']}"
    with st.expander(label):
        st.write(documents['documents'][i])
```

## Strona główna

Na koniec przygotujmy faktyczny interfejs chatu. Na początku dajmy możliwość wybrania 
języka prompta oraz modelu.

```python
with st.sidebar:
    templates = get_language_templates()
    template_key = st.selectbox("Select prompt language", templates.keys())
    template = templates[template_key]

    model_name = st.selectbox("Select model", ["gpt-3.5-turbo", "gpt-4"])
    model = ChatOpenAI(temperature=0.0, model=model_name)
```

Teraz możemy obsłużyć interakcję z użytkownikiem. Na początku pobieramy pytanie, następnie
budujemy łańcuch zapytań. Na koniec wyświetlamy odpowiedź.

Używamy łańcucha który opracowaliśmy w poprzednim etapie. Dzięki temu możemy wyświetlić
użytkownikowi jakie dokumenty zostały użyte do wygenerowania odpowiedzi.

```python
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
```

## Podsumowanie

W tym etapie nauczyliśmy się jak używać Streamlit do tworzenia interfejsów graficznych.
Dzięki temu możemy łatwo testować nasze modele i prototypować pomysły.
A co najważniejsze, możemy pokazać nasze modele innym osobom w przyjazny dla nich sposób.

