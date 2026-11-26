"""Спільні функції RAG-пайплайну (ЛР3): чанкінг, ембединги, ChromaDB, пошук і генерація."""
import os
import sys

import chromadb
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import EMBED_MODEL, MODEL, get_client  # noqa: E402

client = get_client()


def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


def embed(texts: list[str], task_type: str) -> list[list[float]]:
    """Ембединги пакетами по 50 текстів (обмеження API на один запит)."""
    vectors = []
    for i in range(0, len(texts), 50):
        res = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 50],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in res.embeddings)
    return vectors


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


def answer(query: str, context: str, topic: str = 'робочій програмі курсу "Prompt інжиніринг"') -> str:
    final_prompt = (SYSTEM_PROMPT.format(topic=topic) +
                    f"\n**Контекст:**\n{context}\n\n**Запитання:**\n{query}\n")
    response = client.models.generate_content(model=MODEL, contents=final_prompt)
    return response.text.strip()
