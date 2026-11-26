"""Лабораторна робота №4. Кейс 1 (вартість) і Кейс 2 (швидкодія).

Порівнюються дві моделі: основна (умовно «Pro») і легша (умовно «Flash»).
Назви моделей задаються змінними GEMINI_MODEL і GEMINI_FLASH_MODEL (див. gemini_setup.py).
Ціни умовні, за методичними вказівками (USD за 1 млн токенів):
  Pro:   $1.00 вхідні, $3.00 вихідні
  Flash: $0.10 вхідні, $0.30 вихідні
"""
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import FLASH_MODEL, MODEL, get_client  # noqa: E402

client = get_client(fallback=False)  # вимірюємо саме задані моделі, без резервної
model_pro, model_flash = MODEL, FLASH_MODEL
PRICES = {model_pro: (1.00, 3.00), model_flash: (0.10, 0.30)}
RUNS = int(os.getenv("RUNS", "3"))

# Контекст, який «знайшла» RAG-система: 3 абзаци з робочої програми курсу
context = (
    "Метою вивчення навчальної дисципліни «PROMPT інжиніринг» є формування у здобувачів освіти "
    "фундаментальних знань та практичних навичок з проєктування, розробки та інтеграції великих мовних "
    "моделей (LLM) у сучасні програмні системи та продукти.\n"
    "Навчальна дисципліна «PROMPT інжиніринг» забезпечує формування у здобувачів освіти другого "
    "(магістерського) освітнього рівня вищої освіти таких компетентностей: здатність до абстрактного "
    "мислення, аналізу та синтезу; здатність застосовувати знання у практичних ситуаціях; здатність "
    "вчитися й оволодівати сучасними знаннями; здатність формалізувати предметну область певного проєкту "
    "у вигляді відповідної інформаційної моделі; здатність розробляти, описувати, аналізувати та "
    "оптимізувати архітектурні рішення інформаційних та комп'ютерних систем різного призначення; "
    "здатність розробляти програмне забезпечення відповідно до сформульованих вимог з урахуванням наявних "
    "ресурсів та обмежень.\n"
    "У результаті успішного завершення курсу здобувачі вищої освіти сфокусовано розвинуть ключові "
    "інженерні навички, необхідні для роботи з передовими технологіями штучного інтелекту. Курс "
    "концентрується на формуванні здатності до створення інноваційних програмних рішень, від розробки "
    "концептуальної моделі та архітектури до безпосередньої реалізації."
)
query = "Яка мета курсу 'Prompt інжиніринг'?"
final_prompt = f"Контекст: {context}\n\nЗапитання: {query}\n\nВідповідь:"


def cost(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = PRICES[model]
    return tokens_in / 1e6 * p_in + tokens_out / 1e6 * p_out


def case1():
    print("=" * 72)
    print("КЕЙС 1. Аналіз вартості")
    print(f"Моделі: «Pro» = {model_pro}, «Flash» = {model_flash}; ціни умовні (методичні вказівки)")
    tin = {m: client.models.count_tokens(model=m, contents=final_prompt).total_tokens for m in PRICES}
    for m in PRICES:
        print(f"Кількість вхідних токенів для {m}: {tin[m]}")
    out = 100  # припущення з методички: відповідь ~100 токенів
    c = {m: cost(m, tin[m], out) for m in PRICES}
    for m in PRICES:
        p_in, p_out = PRICES[m]
        print(f"Вартість запиту до {m} (вихід {out} ток.): {tin[m]}×${p_in}/1M + {out}×${p_out}/1M = ${c[m]:.7f}")
    saving = (1 - c[model_flash] / c[model_pro]) * 100
    print(f"Flash дешевша за Pro на {saving:.1f}% (у {c[model_pro] / c[model_flash]:.1f} раза)")
    print(f"Для 10 000 таких запитів: Pro ${c[model_pro] * 1e4:.2f}, Flash ${c[model_flash] * 1e4:.2f}")
    return tin


def case2():
    print("=" * 72)
    print(f"КЕЙС 2. Аналіз швидкодії ({RUNS} запуски на модель)")
    stats = {}
    for m in PRICES:
        times, usage = [], None
        for i in range(RUNS):
            start = time.time()
            response = client.models.generate_content(model=m, contents=final_prompt)
            elapsed = time.time() - start
            times.append(elapsed)
            usage = response.usage_metadata
            print(f"  {m}, запуск {i + 1}: {elapsed:.2f} с "
                  f"(вихід {usage.candidates_token_count} ток., міркування {usage.thoughts_token_count or 0} ток.)")
        out_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        stats[m] = (statistics.mean(times), min(times), max(times), usage.prompt_token_count, out_tokens)
        print(f"Час відповіді {m}: середній {stats[m][0]:.2f} с (мін {stats[m][1]:.2f}, макс {stats[m][2]:.2f})")
    print("\nПідсумкова таблиця")
    print(f"{'Модель':24}{'Середній час, с':>16}{'Вхід, ток.':>12}{'Вихід*, ток.':>14}{'Фактична ціна, $':>18}")
    for m, (avg, mn, mx, tin, tout) in stats.items():
        print(f"{m:24}{avg:>16.2f}{tin:>12}{tout:>14}{cost(m, tin, tout):>18.7f}")
    print("* вихідні токени разом з токенами міркування (тарифікуються як вихідні)")
    fast = min(stats, key=lambda k: stats[k][0])
    print(f"Швидша модель: {fast}; різниця середнього часу {abs(stats[model_pro][0] - stats[model_flash][0]):.2f} с")
    print("\nВідповідь Flash (останній запуск):")
    print(response.text.strip()[:600])


if __name__ == "__main__":
    case1()
    case2()
