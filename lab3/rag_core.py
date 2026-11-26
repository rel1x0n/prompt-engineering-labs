"""Спільні функції RAG-пайплайну (ЛР3): чанкінг, ембединги, ChromaDB, пошук і генерація."""
import hashlib
import json
import os
import re
import sys
import time

import chromadb
from google.genai import errors, types
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import EMBED_MODEL, MODEL, get_client  # noqa: E402

client = get_client()


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".cache", "embeddings.json")


def _load_cache() -> dict:
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _embed_batch(batch: list[str], task_type: str):
    """Один запит до API; при перевищенні хвилинної квоти (429) чекаємо й повторюємо."""
    for attempt in range(4):
        try:
            return client.models.embed_content(model=EMBED_MODEL, contents=batch,
                                               config=types.EmbedContentConfig(task_type=task_type))
        except errors.ClientError as e:
            if e.code != 429 or attempt == 3:
                raise
            m = re.search(r"retry in ([\d.]+)s", str(e))
            wait = min(float(m.group(1)) + 2 if m else 30, 70)
            print(f"[квоту ембедингів на хвилину вичерпано, повтор через {wait:.0f} с]", file=sys.stderr)
            time.sleep(wait)


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Ембединги пакетами по 50 текстів (обмеження API на один запит).

    Вектори кешуються на диску (ключ — модель, тип задачі й текст), тому повторна індексація
    того самого документа не витрачає квоту API.
    """
    cache = _load_cache()
    keys = [hashlib.sha1(f"{EMBED_MODEL}|{task_type}|{t}".encode()).hexdigest() for t in texts]
    missing = [i for i, k in enumerate(keys) if k not in cache]
    for start in range(0, len(missing), 50):
        idx = missing[start:start + 50]
        res = _embed_batch([texts[i] for i in idx], task_type)
        for i, e in zip(idx, res.embeddings):
            cache[keys[i]] = e.values
    if missing:
        _save_cache(cache)
    return [cache[k] for k in keys]


def build_collection(chunks: list[str], name: str, metadatas=None):
    db = chromadb.Client()  # in-memory база даних
    collection = db.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    embeddings = embed(chunks, "RETRIEVAL_DOCUMENT")
    collection.add(ids=[str(i) for i in range(len(chunks))], embeddings=embeddings,
                   documents=chunks, metadatas=metadatas)
    return collection, embeddings


def retrieve(collection, query: str, n_results: int = 3):
    q_emb = embed([query], "RETRIEVAL_QUERY")[0]
    res = collection.query(query_embeddings=[q_emb], n_results=n_results)
    return res["ids"][0], res["documents"][0], res["distances"][0], (res.get("metadatas") or [[None]])[0]


SYSTEM_PROMPT = """Ти - чат-бот-консультант по {topic}.
Твоя задача - чітко відповідати на питання студента, базуючись **виключно** на наданому контексті.
Якщо відповіді немає в контексті, так і скажи: "На жаль, у наданих матеріалах немає інформації на це питання."
"""

# Покращена версія (самостійне завдання 1): той самий принцип «лише контекст», але модель
# не відкидає відповідь через інше формулювання питання і чесно повідомляє про часткову інформацію.
SYSTEM_PROMPT_V2 = """Ти - чат-бот-консультант по {topic}.
Правила відповіді:
1. Використовуй лише наданий контекст, без власних знань.
2. Питання може бути сформульоване іншими словами, ніж текст програми (синоніми, ширша або вужча
   назва розділу). Якщо в контексті є відомості, які відповідають на питання по суті, - використай їх.
3. Якщо в контексті є лише частина потрібної інформації, наведи її і прямо вкажи, чого в матеріалах немає.
4. Якщо в контексті немає нічого, що стосується питання, відповідай рівно:
   "На жаль, у наданих матеріалах немає інформації на це питання."
5. Наприкінці вкажи розділ робочої програми, на який спирається відповідь.
"""

PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2}


def answer(query: str, context: str, topic: str = 'робочій програмі курсу "Prompt інжиніринг"',
           version: str = "v1") -> str:
    final_prompt = (PROMPTS[version].format(topic=topic) +
                    f"\n**Контекст:**\n{context}\n\n**Запитання:**\n{query}\n")
    response = client.models.generate_content(model=MODEL, contents=final_prompt)
    return response.text.strip()
