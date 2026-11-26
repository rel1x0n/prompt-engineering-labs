# Prompt-інжиніринг — лабораторні роботи

Рудько Юрій, група мКН-24, фізико-математичний факультет, ТНПУ ім. В. Гнатюка.

Код лабораторних робіт, самостійної роботи та підсумкового проєкту з дисципліни «Prompt-інжиніринг»
(спеціальність «Комп'ютерні науки»). Використано офіційний Python SDK **google-genai** і моделі Gemini з Google AI Studio.

## Структура

| Папка | Зміст |
|---|---|
| `lab1/` | ЛР1: промпти Few-Shot / CoT / ToT та результати (Google AI Studio) |
| `lab2/` | ЛР2: ініціалізація, текстовий запит, Function Calling, JSON-план уроку, калькулятор-агент |
| `lab3/` | ЛР3: RAG над робочою програмою (`knowledge.txt`), ChromaDB, база знань з кількох файлів |
| `lab4/` | ЛР4: вартість і затримка Pro vs Flash, streaming, атака/захист prompt injection |
| `lab1_preparation/` … `lab4_preparation/` | Самостійна робота до кожної ЛР |
| `final_project/` | Підсумковий проєкт TestGen — генератор unit-тестів pytest на базі Gemini |
| `self_work/` | Colab-блокнот для запуску скриптів самостійної роботи |

## Запуск локально

```bash
pip install -r requirements.txt
cp .env.example .env        # вписати власний ключ GEMINI_API_KEY
python lab2/main.py
python lab3/main.py "Скільки годин відведено на лабораторні роботи?"
python final_project/testgen.py final_project/examples/grades.py
```

## Запуск у Google Colab

Ключ додається в **Colab → Secrets** під назвою `GEMINI_API_KEY` (доступ для блокнота увімкнено).
Блокноти `lab*/lab*_colab.ipynb` клонують репозиторій, встановлюють залежності й запускають скрипти:

* [ЛР2](https://colab.research.google.com/github/rel1x0n/prompt-engineering-labs/blob/main/lab2/lab2_colab.ipynb)
* [ЛР3](https://colab.research.google.com/github/rel1x0n/prompt-engineering-labs/blob/main/lab3/lab3_colab.ipynb)
* [ЛР4](https://colab.research.google.com/github/rel1x0n/prompt-engineering-labs/blob/main/lab4/lab4_colab.ipynb)
* [Самостійна робота](https://colab.research.google.com/github/rel1x0n/prompt-engineering-labs/blob/main/self_work/self_work_colab.ipynb)
* [Підсумковий проєкт](https://colab.research.google.com/github/rel1x0n/prompt-engineering-labs/blob/main/final_project/final_project_colab.ipynb)

## Моделі

Назви моделей задаються змінними оточення (значення за замовчуванням — у `gemini_setup.py`):
`GEMINI_MODEL` (основна модель, `gemini-3.5-flash-lite`), `GEMINI_FLASH_MODEL` (легша модель, `gemini-3.1-flash-lite`),
`GEMINI_EMBED_MODEL` (ембединги, `gemini-embedding-001`). Якщо модель тимчасово перевантажена (503) або вичерпано
квоту (429), клієнт з `get_client()` повторює запит з резервною моделлю (`GEMINI_FALLBACK_MODELS`).
Моделі Gemini 1.5, згадані в методичних вказівках, виведено з обігу, тому використано актуальні моделі Gemini
та SDK `google-genai` замість застарілого `google-generativeai`.

## Безпека

Ключ API не зберігається в коді: `.env` додано до `.gitignore`, у Colab використовуються Secrets.
