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
Ustawienie temperatury na 0 spowoduje, że model będzie generował w miarę determinstyczne odpowiedzi.
Domyślnie nasz model używa modelu GPT-3.5 Turbo. Możemy zmienić na GPT-4, ale jest on droższy, więc przy
prototypowaniu zostańmy przy GPT-3.5 Turbo. Na koniec warsztaów możemy użyć GPT-4, aby zobaczyć jakie są różnice.

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

Użyjmy naszego kodu z poprzeedniego etapu, aby wyciągnąć dokumenty pasujące do pytania.

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

Dobrze, ponieważ mamy już dokumenty zaindeksowane, to możemy zapytać model o coś i wygenerować odpowiedź na podstawie tych dokumentów. Spytajmy o to jak odbywa się trening modeli LLM.

Na początku poszukajmy najbardziej pasujących dokumentów.

```python
question = "Opisz jak odbywa się trening modeli LLM."
documents = db.similarity_search(question, k=3)
print("Most similar documents:")
for doc in documents:
    print(doc.page_content[:100] + '...')
```

Teraz musimy przygotować pytanie do modelu. Musimy dodać kontekst, aby model mógł wygenerować odpowiedź na podstawie kontekstu. Zapiszmy dokumenty jako pojedyńczy tekst.

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

Wyświeltmy prompt i odpytajmy model aby wygenerował odpowiedź.

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

