# TestGen — генератор unit-тестів pytest на базі Gemini API

Інструмент приймає Python-модуль і автоматично створює для нього набір тестів pytest.
Агент сам запускає свої тести, виправляє помилки в них і відрізняє їх від реальних багів у коді.

**Цільова аудиторія:** студенти-програмісти, яким потрібно навчитися писати тести, та викладачі, які
перевіряють навчальні модулі.

## Архітектура

```
модуль .py ──► [Gemini: системний промпт v3]
                 │ 1. CoT-аналіз вимог (docstring/коментарі)
                 │ 2. план тестів (класи еквівалентності, межі, винятки)
                 │ 3. код тестів у стилі Few-Shot прикладу
                 ▼
          Function Calling: run_pytest(test_code) ──► пісочниця (tmp-тека, timeout 60 с, pytest + coverage)
                 ▲                                         │
                 └──────────── результати, покриття ◄──────┘   (до 3 ітерацій)
                 ▼
          JSON: test_file + bugs + summary  ──►  generated_tests/test_<module>.py, report_<module>.json
```

## Запуск

```bash
pip install -r requirements.txt
python final_project/testgen.py final_project/examples/grades.py
python final_project/testgen.py final_project/examples/discount_buggy.py   # модуль з навмисними багами
python final_project/compare_prompts.py final_project/examples/grades.py          # порівняння промптів v1/v2/v3
python final_project/compare_prompts.py final_project/examples/discount_buggy.py  # те саме на модулі з багами
```

Моделі задаються в `gemini_setup.py` (за замовчуванням `gemini-3.5-flash-lite`, резервна `gemini-3.1-flash-lite`).

У Colab — блокнот [`final_project_colab.ipynb`](https://colab.research.google.com/github/rel1x0n/prompt-engineering-labs/blob/main/final_project/final_project_colab.ipynb).

## Промпти

* `prompts/system_v1.txt` — наївний промпт (вихідна точка);
* `prompts/system_v2.txt` — роль + Few-Shot приклад стилю;
* `prompts/system_v3.txt` — фінальна версія: CoT-аналіз вимог, Few-Shot, агентний цикл з `run_pytest`,
  правила класифікації невдалих тестів (помилка тесту / баг коду), JSON-формат результату, захист від
  інструкцій, вбудованих у код модуля.

## Обмеження й безпека

* Згенеровані тести виконуються в тимчасовій теці з таймаутом, але без повної ізоляції. Для продакшну
  потрібен контейнер без мережі.
* Код модуля передається в API: не використовувати для коду з секретами чи персональними даними.
* Вивід моделі завжди перевіряється запуском pytest. Пісочниця запускає pytest з `xfail_strict=true`, тому
  «баг», тест якого насправді проходить, не приймається; підсумковий статус (`ok` / `needs_review`) визначає код,
  а не модель.
