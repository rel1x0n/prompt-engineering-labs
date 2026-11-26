"""Запуск промптів з prompts.md через Gemini API (самостійна робота до ЛР1)."""
import json
import os
import re
import sys

from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import MODEL, get_client  # noqa: E402

client = get_client()
HERE = os.path.dirname(os.path.abspath(__file__))
md = open(os.path.join(HERE, "prompts.md"), encoding="utf-8").read()
blocks = re.findall(r"```text\n(.*?)```", md, flags=re.S)
server_prompt, step1, step2, step3, step4 = blocks


def gen(prompt, json_mode=False):
    cfg = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    return client.models.generate_content(model=MODEL, contents=prompt, config=cfg).text.strip()


print("=" * 70, "\nПРОМПТ 1. Конфігурація сервера (вкладений JSON)")
cfg = json.loads(gen(server_prompt, json_mode=True))
print(json.dumps(cfg, ensure_ascii=False, indent=2))
print(f"Перевірка: дисків {len(cfg['hardware']['disks'])}, сервісів {len(cfg['services'])}, "
      f"cron бекапу '{cfg['backup']['schedule_cron']}', зберігання {cfg['backup']['retention_days']} днів")

print("=" * 70, "\nПРОМПТ 2. Ланцюжок обробки відгуку")
review = ("Курс загалом цікавий, але вже третій день не відкриваються відео в модулі 4 — постійно крутиться "
          "завантаження. Також у лекції 3 у прикладі коду помилка в імені змінної. Дедлайн по лабі в п'ятницю, "
          "а я не можу подивитися матеріал!")
analysis = gen(step1.replace("{review}", review), json_mode=True)
print("Крок 1 — аналіз:", analysis)
routing = gen(step2.replace("{analysis_json}", analysis))
print("Крок 2 — маршрутизація:", routing.splitlines()[-1])
reply = gen(step3.replace("{analysis_json}", analysis).replace("{routing}", routing.splitlines()[-1]))
print("Крок 3 — відповідь студенту:\n" + reply)
check = gen(step4.replace("{reply}", reply).replace("{routing}", routing.splitlines()[-1]), json_mode=True)
print("Крок 4 — самоперевірка:", check)
