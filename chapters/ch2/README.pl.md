# Etap 2

Witam na warsztatach poświęconych Generative AI!
W tym etapie dowiemy się jak: 

- wykorzystać model do generowania tekstu
- wyciągnąć dokumenty pasujące do pytania
- wygenerować odpowiedź na pytanie na ich podstawie
- uprościć cały proces

## Generowanie tekstu

W tym etapie wykorzystamy model do generowania tekstu. Na początku dodajmy biblioteki, które będą nam potrzebne.

```python
from langchain.chat_models import ChatOpenAI
import dotenv

dotenv.load_dotenv()
```

Utowrzymy model, który będzie generował tekst na podstawie pytania.

```python
model = ChatOpenAI(temperature=0.0)
```
Ustawienie temperatury na 0 spowoduje, że model będzie generował w miarę deterministyczne odpowiedzi.
Domyślnie nasz model używa modelu GPT-3.5 Turbo. Możemy zmienić na GPT-4, ale jest on droższy, więc zostańmy przy GPT-3.5 Turbo. Na koniec warsztatów możemy użyć GPT-4, aby zobaczyć jakie są różnice.

Wywołajmy model z użyciem metody `invoke`:

```python
print(model.invoke("Przywitaj się z uczestnikami naszego kursu."))
print(model.invoke("Przywitaj się z uczestnikami naszego kursu."))
```

co daje nam:

```
content='Witajcie uczestnicy naszego kursu! Jestem tu, aby pomóc wam w nauce i rozwijaniu waszych umiejętności. Mam nadzieję, że będziecie mieć owocne i satysfakcjonujące doświadczenie podczas naszych zajęć. Chętnie poznam was wszystkich i wspólnie przejdziemy przez ten kurs. Dajcie znać, jeśli macie jakiekolwiek pytania lub potrzebujecie pomocy. Powodzenia i bawcie się dobrze!'
content='Witajcie uczestnicy naszego kursu! Jestem tu, aby pomóc wam w nauce i rozwijaniu waszych umiejętności. Mam nadzieję, że będziecie mieć owocne doświadczenie i czerpać z tego kursu jak najwięcej. Jeśli macie jakiekolwiek pytania, nie wahajcie się ich zadawać. Razem stworzymy inspirującą i efektywną przestrzeń nauki. Powodzenia i bawcie się dobrze!'
```

Na razie nasz model generuje tekst na podstawie wejścia. Możemy go zapytać o coś, ale nie będzie posiadał kontekstu.

Spróbujmy dodać kontekst!

## Wyciąganie dokumentów i generowanie odpowiedzi

W tym etapie wykorzystamy model do wyciągnięcia dokumentów pasujących do pytania i wygenerowania odpowiedzi na podstawie tych dokumentów.

Użyjmy naszego kodu z poprzedniego etapu, aby wyciągnąć dokumenty pasujące do pytania.

```python
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from pathlib import Path

file_path = Path(__file__).parent / "blog-entry.pl.pdf"
loader = PyPDFLoader(str(file_path))

chunks = loader.load_and_split()

db = Chroma(embedding_function=OpenAIEmbeddings())
chunk_ids = db.add_documents(chunks)
```

Dobrze, ponieważ mamy już dokumenty zindeksowane, to możemy zapytać model o coś i wygenerować odpowiedź na podstawie tych dokumentów. Spytajmy o to jak odbywa się trening modeli LLM.

Na początku poszukajmy najbardziej pasujących dokumentów.

```python
question = "Opisz jak odbywa się trening modeli LLM."
documents = db.similarity_search(question, k=3)
print("Most similar documents:")
for doc in documents:
    print(doc.page_content[:100] + '...')
```

Teraz musimy przygotować pytanie do modelu. Musimy dodać kontekst, aby model mógł wygenerować odpowiedź na podstawie kontekstu. Zapiszmy dokumenty jako pojedynczy tekst.

```python
context = "\n\n".join([d.page_content for d in documents])
```

Teraz możemy przygotować prompt który wyślemy do modelu:

```python
prompt = f"""Odpowiedź na pytanie używając tylko następującego kontekstu: 

{context}

Pytanie: {question}
"""
```

Wyświetlmy prompt i odpytajmy model aby wygenerował odpowiedź.

```python
print()
print("----- Prompt: -----")
print(prompt)

answer = model.invoke(prompt)
print()
print("----- Answer: -----")
print(answer)
```

A o to przykładowa odpowiedź naszego modelu:

```
----- Answer: -----
content='Trening modeli LLM odbywa się w dwóch etapach: treningu wstępnym i dostrajaniu. W treningu wstępnym model uczy się ogólnych zasad języka poprzez zgadywanie kolejnych części tekstu na podstawie poprzedzającego fragmentu. W wyniku tego etapu otrzymujemy model bazowy. Następnie, w etapie dostrajania, model jest ćwiczony w sensownym odpowiadaniu na pytania. Podczas treningu model bazowy jest dzielony na jednostki zwane tokenami, które są fragmentami tekstu. Trening na zbiorze dialogów polega na używaniu starannie dobranych przykładów, składających się z par podpowiedź-uzupełnienie. W tym etapie model jest trenowany, aby zgadywać kolejne tokeny. Trening modeli LLM wymaga większych nakładów pracy, ponieważ wymaga zgromadzenia dużej ilości dialogów, które są używane jako dane treningowe.'
```

## Uproszczenie procesu

Cały process jest bardzo prosty, ale nadal nie dowiedzieliśmy się czym jest *chain* w nazwie *langchain*. Przyjrzyjmy się temu w bliżej.

Chain to inaczej łańcuch, łańcuch połączonych ze sobą operacji. Elementy w łańcuchu są połączone ze sobą i każdy kolejny element łańcucha używa wyniku poprzedniego elementu. W naszym przypadku łańcuch będzie wyglądał. Elementy łańcucha posiadają metodę `invoke`.
Gdzieś już ją widzieliśmy - model LLM udostępniony przez `langchain` jest właśnie takim elementem łańcucha.

Ale, nasz prompt nie posiada metody invoke, ponieważ jest to tylko tekst. Musimy go przekształcić w element łańcucha. Zróbmy to!

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

template = """Odpowiedź na pytanie używając tylko następującego kontekstu:

{context}

Pytanie: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
result = prompt.invoke({"context": "Wypełnimy później", "question": question})

print(type(result), result)
```

Zobaczmy co dostaliśmy:
```
<class 'langchain.prompts.chat.ChatPromptValue'> messages=[HumanMessage(content='Odpowiedź na pytanie używając tylko następującego kontekstu:\n\nWypełnimy później\n\nPytanie: Opisz jak odbywa się trening modeli LLM.\n')]
```

Jak widzimy nie mamy już tylko tekstu, a obiekt `ChatPromptValue`.
Zawiera on w sobie listę wiadomości, w tym wypadku tylko jedną. Wiadomość od człowieka, która zawiera prompt, który wyślemy do modelu. Dzięki temu możemy rozróżnić wiadomości od człowieka i od modelu.

No tak, ale nadal musimy ręcznie wypełnić informacje o kontekście. Niestety, ale
Chroma nie posiada metody `invoke`, więc nie możemy go użyć jako element łańcucha.
Ale posiada metodę `as_retriever`, a retriever posiada metodę `invoke`. Zróbmy to!


```python
retriever = db.as_retriever()
result = retriever.invoke(question)
print(type(result), result)
```

A oto wynik:
```
<class 'list'> [Document(page_content='Trenowanie dużych \n modeli językowych \n Źródło:  https://airrival.pl/trenowanie-duzych-modeli-jezykowych/ \n Rozpoczynamy cykl artykułów pod tytułem  Anatomia LLM  ...
```
i tak dalej. Jak widzimy dostajemy listę dokumentów, które pasują do pytania. 
Czyli ostatecznie dostaliśmy podobny wynik do tego, który dostaliśmy w poprzednim etapie.
Jednak teraz możemy użyć tego jako element łańcucha.

Spróbujmy połączyć wszystkie elementy w jeden element. To jest główna siła langchain.
Komponowanie prostych elementów w bardziej złożone. Ponieważ łańcuch posiada `invoke`
oraz elementy łańcucha również posiadają `invoke`, to możemy zagnieżdżać łańcuchy w sobie.

```python
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
```

Trochę się tutaj dzieje, więc odpakujmy to po kolei:
* utworzylismy funkcję `format_docs`, która przyjmuje listę dokumentów i zwraca tekst. Funkcje można używać jako elementy łańcucha.
* `retriever | format_docs` to łańcuch który najpierw wywoła retriever, a następnie wywoła funkcję `format_docs` z wynikiem retrievera.
* `RunnablePassthrough()` to sposób na dostarczenie parametrów z zewnątrz. Tak na prawdę przekażemy tutaj parametr `question`, który jest pytaniem.
* `{"context": retriever | format_docs, "question": RunnablePassthrough()}` to słownik który zaweira łańcucy o nazwie: `context` oraz `question`.
* ostatecznie łączymy teń łańcuch z elementem `prompt` który wygeneruje zapytanie do modelu

Dodajmy teraz model do łańcucha:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
)

print()
print("----- Chain with model: -----")
print(chain.invoke(question))
```

Jak widzicie dodanie modelu jest banalne, jest to po prostu kolejny element łańcucha.
Tutaj mamy przykład wyjścia:
```
----- Chain with model: -----
content='Trening modeli LLM odbywa się w dwóch etapach: treningu wstępnym i dostrajaniu. W treningu wstępnym model uczy się ogólnych zasad języka poprzez zgadywanie kolejnych części tekstu na podstawie poprzedzającego go fragmentu. W tym etapie tworzony jest model bazowy. Następnie, w etapie dostrajania, model jest ćwiczony w sensownym odpowiadaniu na pytania. Trening na zbiorze dialogów polega na używaniu starannie dobranych przykładów dialogów, składających się z par podpowiedź-uzupełnienie. W tym etapie model jest trenowany, aby generował odpowiedzi na podstawie podanych pytań. W celu oceny generowanych tekstów, tworzona jest hierarchia rankingowa, która jest następnie wykorzystywana do trenowania modelu nagradzania. Ostateczny trening odbywa się poprzez generowanie kilku alternatywnych uzupełnień podpowiedzi i wybór najlepszego tekstu na podstawie oceny modelu nagradzania.'
```

Zwrócona odpowiedź nie jest zwykłym tekstem, a odpowiedzią modelu. Możemy użyć jeszcze kolejnego
elementu łańcucha, który zmieni odpowiedź modelu na tekst.

```python
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
```

I odpowiedź:
```
----- Chain with model and parser: -----
Trening modeli LLM odbywa się w dwóch etapach: treningu wstępnym i dostrajaniu. W treningu wstępnym model uczy się ogólnych zasad języka poprzez zgadywanie kolejnych części tekstu na podstawie poprzedzającego go fragmentu. W tym etapie tworzony jest model bazowy. Następnie, w etapie dostrajania, model jest ćwiczony w sensownym odpowiadaniu na pytania. Trening na zbiorze dialogów polega na używaniu starannie dobranych przykładów, składających się z par podpowiedź-uzupełnienie. W tym etapie model jest trenowany na podstawie rankingów ocenianych przez ludzi. Ostateczny trening odbywa się poprzez współdziałanie modeli, generując kilka alternatywnych uzupełnień podpowiedzi i wybierając najlepszy tekst na podstawie oceny modelu nagradzania.
```

Bardzo wygodne, prawda?

## Łańcuch zwracający użyte dokumenty

Na koniec przetransformujmy nasz łańcuch tak, aby zwracał również użyte dokumenty.

```python
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
```

Używamy tutaj elementu `RunnableParallel`, który pozwala na uruchomienie kilku elementów łańcucha równolegle. Ten łańcuch jest bardziej złożony niż poprzedni, ale nadal jest czytelny:

* `RunnableParallel({"documents": retriever, "question": RunnablePassthrough()})` - tworzymy
łańcuch który wyszuka dokumenty i przechwyci nasze zapytanie z w wejścia.
* Następnie przekazujemy do kolejnego złożonego łańcucha używając `|`.
`"documents"` jest łańcuchem który wyciągnie metadane z dokumentów, a `"answer"` jest łańcuchem który zwróci odpowiedź modelu. 
*`query_chain` jest łańcuchem który zwróci odpowiedź modelu. Jest on prawie identyczny jak poprzedni łańcuch, jedyna różnica to że operuje na wyjściu naszego łańcucha.

## Podsumowanie

Dużo prototypowania w tym etapie, dlatego zbierzmy cały kod w jedno miejsce:

```python
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

```

