"""ЛР3, самостійне завдання 2: RAG над базою знань з кількох текстових файлів.

Скрипт читає всі .txt з папки lab3_preparation/knowledge, об'єднує їх в один текст
(перед кожним файлом — заголовок з назвою джерела), індексує і відповідає на питання,
відповіді на які містяться в різних файлах.
"""
import glob
import os
import re

from rag_core import answer, build_collection, retrieve, split_text

HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "..", "lab3_preparation", "knowledge")

QUESTIONS = [
    "Хто і в якому році запропонував архітектуру RAG і які компоненти використовувались в оригінальній роботі?",
    "Що таке Tree-of-Thought і чим він відрізняється від Chain-of-Thought?",
    "Чому непряма prompt injection небезпечніша за пряму?",
    "Яка розмірність векторів gemini-embedding-001 і які типи індексів підтримує FAISS?",
    "Яка ціна підписки на Gemini Advanced?",  # відповіді немає в базі знань
]


def load_files():
    parts, names = [], []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.txt"))):
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            parts.append(f"=== Джерело: {name} ===\n{f.read().strip()}")
        names.append(name)
    return "\n\n".join(parts), names


def source_of(chunk: str, full_text: str) -> str:
    """Визначає файл-джерело чанка за найближчим попереднім заголовком «=== Джерело: ... ===»."""
    pos = full_text.find(chunk[:80])
    heads = [(m.start(), m.group(1)) for m in re.finditer(r"=== Джерело: (.+?) ===", full_text)]
    src = [name for start, name in heads if start <= pos]
    return src[-1] if src else "?"


def main():
    text, names = load_files()
    print(f"Завантажено файлів: {len(names)} -> {', '.join(names)}; разом {len(text):,} символів")
    chunks = split_text(text, chunk_size=800, chunk_overlap=100)
    metas = [{"source": source_of(c, text)} for c in chunks]
    collection, _ = build_collection(chunks, "multi_file_kb", metadatas=metas)
    print(f"Чанків: {len(chunks)}; проіндексовано у ChromaDB: {collection.count()}")
    for q in QUESTIONS:
        print("\n" + "=" * 70)
        print("Запит:", q)
        ids, docs, dists, mds = retrieve(collection, q, n_results=3)
        for cid, dist, md in zip(ids, dists, mds):
            print(f"  знайдено: чанк #{cid} з файлу {md['source']} (відстань {dist:.3f})")
        print("Відповідь ШІ:")
        print(answer(q, "\n\n".join(docs), topic="базі знань з промпт-інженерії, RAG і безпеки LLM"))


if __name__ == "__main__":
    main()
