"""Лабораторна робота №3. Базовий RAG-застосунок над робочою програмою курсу.

Запуск:  python lab3/main.py "Запитання 1" ["Запитання 2" ...]
Індексація виконується один раз, далі для кожного запитання — пошук і генерація.
"""
import os
import sys

from rag_core import EMBED_MODEL, MODEL, answer, build_collection, retrieve, split_text

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUERY = "Скільки всього годин відведено на лабораторні роботи на денній формі?"


def main(queries):
    # --- ЗАВАНТАЖЕННЯ ДАНИХ ---
    with open(os.path.join(HERE, "knowledge.txt"), encoding="utf-8") as f:
        knowledge_text = f.read()
    print(f"[1] Дані завантажено: {len(knowledge_text):,} символів.")

    # --- РОЗБИТТЯ НА ЧАНКИ ---
    chunks = split_text(knowledge_text, chunk_size=1000, chunk_overlap=100)
    print(f"[2] Документ розбито на {len(chunks)} чанків (chunk_size=1000, overlap=100).")

    # --- ЕМБЕДИНГИ + ВЕКТОРНА БД ---
    collection, embeddings = build_collection(chunks, "course_syllabus")
    print(f"[3] Створено {len(embeddings)} векторних представлень (модель {EMBED_MODEL}, "
          f"розмірність {len(embeddings[0])}).")
    print(f"[4] Векторну базу даних ChromaDB створено та наповнено ({collection.count()} записів).")

    # --- ПОШУК ТА ГЕНЕРАЦІЯ ---
    for query in queries:
        print("\n" + "=" * 70)
        print(f"Запит користувача: {query}")
        ids, docs, dists, _ = retrieve(collection, query, n_results=3)
        print("\nЗнайдений контекст (3 найближчі чанки):")
        for cid, doc, dist in zip(ids, docs, dists):
            preview = " ".join(doc.split())[:300]
            print(f"  - чанк #{cid} (косинусна відстань {dist:.3f}): {preview}...")
        context = "\n".join(docs)
        print(f"\nВідповідь ШІ ({MODEL}):")
        print(answer(query, context))


if __name__ == "__main__":
    main(sys.argv[1:] or [DEFAULT_QUERY])
