# Порівняльний аналіз векторних баз даних для RAG: ChromaDB vs FAISS

| Критерій | ChromaDB | FAISS |
|---|---|---|
| Розробник, рік | Chroma (стартап), 2022 | Meta AI Research (Facebook AI), 2017 |
| Ліцензія | Apache 2.0 | MIT |
| Тип | Векторна **база даних** (документи + вектори + метадані) | **Бібліотека** індексів для пошуку подібності |
| Мова / API | Python, JavaScript; клієнт-серверний режим через HTTP | C++ з обгорткою Python |
| Режими роботи | in-memory, локальне збереження на диск, окремий сервер, хмара | у пам'яті процесу; збереження індексу у файл вручну (`write_index`) |
| Алгоритми пошуку | HNSW | Flat (точний), IVF, PQ/OPQ, HNSW, LSH та їх комбінації |
| Метрики | l2, cosine, inner product | L2, inner product (cosine — через нормалізацію векторів) |
| GPU | ні | так (CUDA), найшвидший варіант для великих обсягів |
| Масштаб | від тисяч до мільйонів векторів на одному вузлі | до мільярдів векторів (IVF+PQ, GPU) |
| Збереження документів і метаданих | вбудоване | немає — потрібна окрема БД / словник |
| Фільтрація за метаданими | вбудована (`where`, `where_document`) | немає, реалізується вручну |
| Оновлення / видалення | `add`, `update`, `upsert`, `delete` | додавання просте, видалення залежить від типу індексу |
| Інтеграції | LangChain, LlamaIndex, вбудовані функції ембедингів | LangChain, LlamaIndex (як vector store) |
| Поріг входу | дуже низький: 3–4 рядки коду до першого пошуку | вищий: потрібно обрати індекс і зберігати відповідність id → текст |
| Типове застосування | прототипи RAG, навчальні проєкти, невеликі продукти | великі колекції, офлайн-пошук, дослідження, високі вимоги до швидкості |

## Висновок для лабораторної роботи №3

Для навчального RAG над робочою програмою курсу (кілька десятків чанків) краще підходить **ChromaDB**: вона зберігає текст чанків разом з векторами і метаданими (наприклад, назвою файлу-джерела) і працює in-memory без налаштування. **FAISS** має сенс, коли колекція налічує мільйони векторів або потрібен GPU-пошук, але тоді доводиться окремо зберігати документи.

## Мінімальний приклад для порівняння

```python
# ChromaDB
import chromadb
col = chromadb.Client().create_collection("kb", metadata={"hnsw:space": "cosine"})
col.add(ids=ids, embeddings=vectors, documents=chunks)
res = col.query(query_embeddings=[q], n_results=3)        # одразу повертає тексти

# FAISS
import faiss, numpy as np
index = faiss.IndexFlatIP(dim)                              # inner product = cosine для нормалізованих векторів
x = np.array(vectors, dtype="float32"); faiss.normalize_L2(x); index.add(x)
qv = np.array([q], dtype="float32"); faiss.normalize_L2(qv)
D, I = index.search(qv, 3)
found = [chunks[i] for i in I[0]]                            # тексти зберігаємо самі
```

## База знань для лабораторної роботи

Підготовлено три текстові файли (папка `knowledge/`):

1. `01_prompting_techniques.txt` — Few-Shot, CoT, Self-Consistency, ToT, prompt chaining, температура.
2. `02_rag_and_vector_db.txt` — RAG (Lewis et al., 2020), чанкінг, ембединги, ChromaDB, FAISS.
3. `03_llm_security.txt` — пряма й непряма prompt injection, OWASP Top 10 for LLM 2025, методи захисту.

Файли використовує скрипт `lab3/multi_file_rag.py`.
