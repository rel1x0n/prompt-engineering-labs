"""Спільне налаштування клієнта Gemini API для всіх лабораторних робіт.

Ключ API ніколи не записується в код. Порядок пошуку:
1. змінна оточення GEMINI_API_KEY (або GOOGLE_API_KEY);
2. файл .env у корені репозиторію (локальний запуск);
3. Colab Secrets (запуск у Google Colab).
"""
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Моделі можна змінити без правки коду через змінні оточення.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")         # основна модель
FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.1-flash-lite")  # легша й дешевша модель
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

# Резервні моделі: якщо модель тимчасово перевантажена (503) або вичерпано квоту (429),
# запит повторюється з наступною моделлю зі списку.
FALLBACK_MODELS = [m.strip() for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", f"{MODEL},{FLASH_MODEL}").split(",") if m.strip()]
RETRY_CODES = {429, 500, 503, 504}


def load_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:  # Google Colab
        from google.colab import userdata  # type: ignore
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            try:
                return userdata.get(name)
            except Exception:
                pass
    except ImportError:
        pass
    raise SystemExit("Не знайдено ключ API: додайте GEMINI_API_KEY у файл .env або в Colab Secrets.")


def _candidates(model: str) -> list:
    out = [model]
    for m in FALLBACK_MODELS:
        if m not in out:
            out.append(m)
    return out


def _note(model: str, err: errors.APIError) -> None:
    print(f"[модель {model} тимчасово недоступна (код {err.code}), запит повторено з резервною моделлю]",
          file=sys.stderr)


def enable_fallback(client: genai.Client) -> genai.Client:
    """Додає до client.models.generate_content(_stream) перемикання на резервну модель.

    Чати (client.chats) теж використовують client.models.generate_content, тому
    перемикання діє і для них.
    """
    models = client.models
    orig, orig_stream = models.generate_content, models.generate_content_stream

    def generate_content(*, model, contents, config=None, **kwargs):
        last = None
        for m in _candidates(model):
            try:
                return orig(model=m, contents=contents, config=config, **kwargs)
            except errors.APIError as e:
                if e.code not in RETRY_CODES:
                    raise
                last = e
                _note(m, e)
        raise last

    def generate_content_stream(*, model, contents, config=None, **kwargs):
        last = None
        for m in _candidates(model):
            try:
                it = iter(orig_stream(model=m, contents=contents, config=config, **kwargs))
                first = next(it)
            except StopIteration:
                return iter(())
            except errors.APIError as e:
                if e.code not in RETRY_CODES:
                    raise
                last = e
                _note(m, e)
                continue

            def chunks(first=first, it=it):
                yield first
                yield from it
            return chunks()
        raise last

    models.generate_content = generate_content
    models.generate_content_stream = generate_content_stream
    return client


def retry_transient(fn, attempts: int = 6, wait: float = 5.0):
    """Викликає fn(); при тимчасовій помилці API (503, 429, ...) чекає й повторює виклик.

    Використовується у вимірюваннях (ЛР4): час вимірюється всередині fn для кожної спроби окремо,
    тому повтори не спотворюють результат.
    """
    for i in range(attempts):
        try:
            return fn()
        except errors.APIError as e:
            if e.code not in RETRY_CODES or i == attempts - 1:
                raise
            print(f"[тимчасова помилка API {e.code}, повтор через {wait:.0f} с]", file=sys.stderr)
            time.sleep(wait)


def get_client(fallback: bool = True) -> genai.Client:
    """Клієнт Gemini API.

    fallback=False — без перемикання на резервну модель і без прихованих повторів SDK: так
    вимірюється саме задана модель (ЛР4), а тимчасові помилки обробляє retry_transient().
    """
    if not fallback:
        once = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
        return genai.Client(api_key=load_api_key(), http_options=once)
    http = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=3))
    return enable_fallback(genai.Client(api_key=load_api_key(), http_options=http))
