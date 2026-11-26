"""ЛР2, самостійне завдання 2.

Калькулятор-агент з чотирма інструментами: add, multiply, subtract, divide.
Цикл Function Calling обробляє будь-яку кількість послідовних викликів.
"""
import os
import sys

from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import MODEL, get_client  # noqa: E402

client = get_client()


def add(a: float, b: float) -> float:
    """Adds two numbers."""
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiplies two numbers."""
    return a * b


def subtract(a: float, b: float) -> float:
    """Subtracts b from a."""
    return a - b


def divide(a: float, b: float) -> float:
    """Divides a by b. Raises an error when b is zero."""
    if b == 0:
        raise ZeroDivisionError("ділення на нуль")
    return a / b


TOOLS = {f.__name__: f for f in (add, multiply, subtract, divide)}


def fmt(fc) -> str:
    return f"{fc.name}(" + ", ".join(f"{k}={v:g}" for k, v in dict(fc.args).items()) + ")"


def ask(prompt: str) -> str:
    config = types.GenerateContentConfig(
        tools=list(TOOLS.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        system_instruction="Ти калькулятор. Для БУДЬ-ЯКИХ обчислень використовуй надані функції, не рахуй подумки.",
    )
    chat = client.chats.create(model=MODEL, config=config)
    print("=" * 70)
    print("Запит:", prompt)
    response = chat.send_message(prompt)
    while response.function_calls:
        parts = []
        for fc in response.function_calls:
            try:
                result = TOOLS[fc.name](**dict(fc.args))
                payload = {"result": result}
                print(f"  виклик {fmt(fc)} -> {result:g}")
            except Exception as exc:  # помилку теж повертаємо моделі, щоб вона пояснила її користувачу
                payload = {"error": str(exc)}
                print(f"  виклик {fmt(fc)} -> помилка: {exc}")
            parts.append(types.Part.from_function_response(name=fc.name, response=payload))
        response = chat.send_message(parts)
    print("Відповідь моделі:", response.text.strip())
    return response.text


if __name__ == "__main__":
    ask("Скільки буде 100 поділити на 4?")
    ask("Від 250 відніми 75, а результат поділи на 7.")
    ask("Скільки буде 15 поділити на 0?")
