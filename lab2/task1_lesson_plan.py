"""ЛР2, самостійне завдання 1.

Промпт змушує модель згенерувати план-конспект уроку з інформатики у форматі JSON,
а скрипт розбирає JSON і виводить окремі елементи в консоль.
"""
import json
import os
import re
import sys

from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import MODEL, get_client  # noqa: E402

client = get_client()

PROMPT = """Ти — методист з інформатики. Склади короткий план-конспект уроку для 8 класу
на тему «Цикли for і while у Python» (45 хвилин).

Поверни ЛИШЕ валідний JSON-об'єкт без пояснень і без markdown, точно за схемою:
{
  "topic": "тема уроку",
  "grade": 8,
  "duration_min": 45,
  "goal": "мета уроку одним реченням",
  "learning_outcomes": ["очікуваний результат 1", "..."],
  "equipment": ["обладнання / ПЗ", "..."],
  "stages": [
    {"name": "назва етапу", "minutes": 5, "activity": "що роблять учитель і учні"}
  ],
  "homework": "домашнє завдання"
}
Вимоги: 4–6 етапів, сума хвилин у stages дорівнює duration_min."""


def parse_json(text: str) -> dict:
    """Розбір JSON з підстраховкою: прибираємо можливу markdown-обгортку ```json ... ```."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(cleaned)


def main():
    response = client.models.generate_content(
        model=MODEL,
        contents=PROMPT,
        # просимо API повернути саме JSON (MIME-тип відповіді)
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    plan = parse_json(response.text)

    print("Тема уроку:", plan["topic"])
    print(f"Клас: {plan['grade']}; тривалість: {plan['duration_min']} хв")
    print("Мета:", plan["goal"])
    print("Очікувані результати:")
    for item in plan["learning_outcomes"]:
        print("  •", item)
    print("Обладнання:", ", ".join(plan["equipment"]))
    print("Етапи:")
    total = 0
    for i, st in enumerate(plan["stages"], 1):
        total += st["minutes"]
        print(f"  {i}. {st['name']} ({st['minutes']} хв) — {st['activity']}")
    print("Домашнє завдання:", plan["homework"])
    status = "OK" if total == plan["duration_min"] else "НЕ збігається"
    print(f"\nПеревірка: сума хвилин етапів = {total} ({status})")


if __name__ == "__main__":
    main()
