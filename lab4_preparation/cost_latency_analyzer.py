"""Самостійна робота до ЛР4: аналізатор вартості та затримки серії API-викликів.

Приймає список промптів (за замовчуванням вбудований, або файл з промптами по одному на рядок),
надсилає кожен до двох моделей і виводить порівняльну таблицю: токени, вартість, час.
Ціни задаються в PRICES (USD за 1 млн токенів) і можуть бути змінені під актуальний прайс.

Запуск: python lab4_preparation/cost_latency_analyzer.py [prompts.txt]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import FLASH_MODEL, MODEL, get_client  # noqa: E402

client = get_client(fallback=False)  # вимірюємо саме задані моделі
PRICES = {MODEL: (1.00, 3.00), FLASH_MODEL: (0.10, 0.30)}  # умовні ціни з методичних вказівок
DEFAULT_PROMPTS = [
    "Поясни одним абзацом, що таке токен у LLM.",
    "Напиши функцію Python, що перевіряє, чи є рядок паліндромом, з docstring.",
    "Склади 5 тестових запитань з відповідями на тему RAG.",
]


def analyze(prompts):
    rows = []
    for p_idx, prompt in enumerate(prompts, 1):
        for model, (p_in, p_out) in PRICES.items():
            start = time.perf_counter()
            resp = client.models.generate_content(model=model, contents=prompt)
            latency = time.perf_counter() - start
            u = resp.usage_metadata
            t_in = u.prompt_token_count or 0
            t_out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
            rows.append((p_idx, model, t_in, t_out, t_in / 1e6 * p_in + t_out / 1e6 * p_out, latency))
    return rows


def print_table(rows):
    hdr = f"{'#':>2}  {'Модель':22}{'Вхід':>7}{'Вихід*':>8}{'Вартість, $':>14}{'Час, с':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r[0]:>2}  {r[1]:22}{r[2]:>7}{r[3]:>8}{r[4]:>14.7f}{r[5]:>9.2f}")
    print("-" * len(hdr))
    for model in PRICES:
        mr = [r for r in rows if r[1] == model]
        print(f"РАЗОМ {model:18}{sum(r[2] for r in mr):>7}{sum(r[3] for r in mr):>8}"
              f"{sum(r[4] for r in mr):>14.7f}{sum(r[5] for r in mr) / len(mr):>9.2f} (сер.)")
    print("* вихідні токени разом з токенами міркування")


if __name__ == "__main__":
    prompts = DEFAULT_PROMPTS
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
    print(f"Промптів: {len(prompts)}; моделі: {', '.join(PRICES)}\n")
    print_table(analyze(prompts))
