"""ЛР4, самостійне завдання 1: звичайна відповідь vs потокова (streaming).

У SDK google-genai потокова генерація викликається методом generate_content_stream
(аналог параметра stream=True у застарілому google-generativeai).
Вимірюються час до першого фрагмента (time to first token, TTFT) і повний час.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import FLASH_MODEL, get_client  # noqa: E402

client = get_client(fallback=False)  # вимірюємо саме задану модель
MODEL = os.getenv("STREAM_MODEL", FLASH_MODEL)
PROMPT = ("Напиши стислий конспект (6–8 абзаців) на тему «Що таке RAG і як побудувати RAG-застосунок "
          "на Python»: етапи індексації, ембединги, векторна БД, пошук і генерація.")


def normal():
    start = time.time()
    response = client.models.generate_content(model=MODEL, contents=PROMPT)
    total = time.time() - start
    print(f"[Звичайний режим] перший текст з'явився через {total:.2f} с (одразу вся відповідь), "
          f"довжина {len(response.text)} символів")
    return total


def streaming():
    start = time.time()
    first, n_chunks, length = None, 0, 0
    print("[Потоковий режим] текст надходить частинами:\n")
    for chunk in client.models.generate_content_stream(model=MODEL, contents=PROMPT):
        if not chunk.text:
            continue
        if first is None:
            first = time.time() - start
        n_chunks += 1
        length += len(chunk.text)
        print(chunk.text, end="", flush=True)
    total = time.time() - start
    print(f"\n\n[Потоковий режим] перший фрагмент через {first:.2f} с, повний час {total:.2f} с, "
          f"фрагментів {n_chunks}, довжина {length} символів")
    return first, total


if __name__ == "__main__":
    print(f"Модель: {MODEL}\n")
    t_normal = normal()
    t_first, t_total = streaming()
    print("\nПорівняння")
    print(f"  Час до появи першого тексту: звичайний {t_normal:.2f} с, потоковий {t_first:.2f} с")
    print(f"  Повний час генерації:        звичайний {t_normal:.2f} с, потоковий {t_total:.2f} с")
    print(f"  Користувач бачить перший текст у {t_normal / t_first:.1f} раза швидше в потоковому режимі")
