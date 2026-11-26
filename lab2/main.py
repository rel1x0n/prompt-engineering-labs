"""Лабораторна робота №2. Робота з LLM API на Python.

Крок 1 — безпечне завантаження ключа та ініціалізація моделі.
Крок 2 — генерація простої текстової відповіді.
Крок 3 — Function Calling: «розумний калькулятор» з функціями add і multiply.

Запуск:  python lab2/main.py [1|2|3]   (без аргументу виконуються всі кроки)
"""
import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import MODEL, enable_fallback, load_api_key  # noqa: E402

# ---------------- Крок 1. Налаштування та ініціалізація ----------------
load_dotenv()                      # змінні оточення з файлу .env
api_key = load_api_key()           # ключ не зберігається в коді
client = enable_fallback(genai.Client(api_key=api_key))


def step1():
    info = client.models.get(model=MODEL)
    print("=" * 70)
    print("КРОК 1. Ініціалізація")
    print(f"Модель: {info.display_name} ({info.name})")
    print(f"Ліміт вхідних токенів: {info.input_token_limit:,}; вихідних: {info.output_token_limit:,}")
    print("Модель успішно ініціалізовано!")


# ---------------- Крок 2. Проста текстова відповідь ----------------
def step2():
    print("=" * 70)
    print("КРОК 2. Текстовий запит")
    prompt = "Поясни концепцію рекурсії в програмуванні, використовуючи аналогію з ляльками-матрьошками."
    print(f"Промпт: {prompt}\n")
    response = client.models.generate_content(model=MODEL, contents=prompt)
    print(response.text)
    u = response.usage_metadata
    print(f"\n[токени: вхідні {u.prompt_token_count}, вихідні {u.candidates_token_count}, "
          f"міркування {u.thoughts_token_count or 0}, усього {u.total_token_count}]")


# ---------------- Крок 3. Function Calling ----------------
def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b


def fmt_call(fc) -> str:
    args = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v!r}" for k, v in dict(fc.args).items())
    return f"{fc.name}({args})"


def run_calculator_chat(prompt: str, tools: dict) -> str:
    """Повний цикл Function Calling з ручною обробкою кожного виклику."""
    config = types.GenerateContentConfig(
        tools=list(tools.values()),
        # вимикаємо автоматичне виконання функцій SDK, щоб показати кожен крок циклу
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    chat = client.chats.create(model=MODEL, config=config)   # сесія чату (аналог start_chat)
    print(f"Запит користувача: {prompt}")
    response = chat.send_message(prompt)
    n = 0
    while response.function_calls:
        label = "Перша відповідь моделі" if n == 0 else "Проміжна відповідь моделі"
        print(f"\n{label}: FunctionCall -> " + "; ".join(fmt_call(fc) for fc in response.function_calls))
        parts = []
        for fc in response.function_calls:
            n += 1
            func = tools[fc.name]
            result = func(**dict(fc.args))           # виконуємо локальну Python-функцію
            print(f"  [{n}] виконано {fmt_call(fc)} = {result:g}")
            parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
        print("  -> результат надіслано моделі (FunctionResponse)")
        response = chat.send_message(parts)
    print(f"\nФінальна текстова відповідь: {response.text}")
    print(f"[викликів функцій: {n}; запитів до API: {n_requests(chat)}]")
    return response.text


def n_requests(chat) -> int:
    # кожне повідомлення користувача/функції в історії = один запит до API
    return sum(1 for c in chat.get_history() if c.role == "user")


def step3():
    print("=" * 70)
    print("КРОК 3. Function Calling")
    run_calculator_chat("Скільки буде, якщо 5 помножити на 12, а потім додати 8?",
                        {"add": add, "multiply": multiply})


if __name__ == "__main__":
    steps = {"1": step1, "2": step2, "3": step3}
    selected = sys.argv[1:] or list(steps)
    for s in selected:
        steps[s]()
