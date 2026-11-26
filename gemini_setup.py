"""Спільне налаштування клієнта Gemini API для всіх лабораторних робіт.

Ключ API ніколи не записується в код. Порядок пошуку:
1. змінна оточення GEMINI_API_KEY (або GOOGLE_API_KEY);
2. файл .env у корені репозиторію (локальний запуск);
3. Colab Secrets (запуск у Google Colab).
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Моделі можна змінити без правки коду через змінні оточення.
MODEL = os.getenv("GEMINI_MODEL", "gemini-4-hydrogen")          # основна («Pro»-клас) модель
FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-3.8-flash")  # швидка модель
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")


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


def get_client() -> genai.Client:
    return genai.Client(api_key=load_api_key())
