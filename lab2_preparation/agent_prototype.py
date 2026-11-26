"""Самостійна робота до ЛР2: агент з двома взаємодіючими інструментами (Function Calling).

Інструмент 1 get_current_date() повертає сьогоднішню дату.
Інструмент 2 get_weather(city, date) повертає прогноз погоди на дату з Open-Meteo (без API-ключа).
Щоб відповісти на «Яка погода буде завтра в Тернополі?», модель має спочатку дізнатися дату,
обчислити «завтра» і лише потім викликати get_weather — інструменти взаємодіють через модель.
"""
import datetime as dt
import os
import sys

import requests
from google.genai import types

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_setup import MODEL, get_client  # noqa: E402

client = get_client()
CITIES = {"тернопіль": (49.5535, 25.5948), "київ": (50.4501, 30.5234), "львів": (49.8397, 24.0297)}


def get_current_date() -> dict:
    """Returns today's date in ISO format (YYYY-MM-DD) and the weekday name."""
    today = dt.date.today()
    return {"date": today.isoformat(), "weekday": today.strftime("%A")}


def get_weather(city: str, date: str) -> dict:
    """Returns the daily weather forecast for a Ukrainian city (Тернопіль, Київ, Львів) on a date YYYY-MM-DD."""
    key = city.strip().lower()
    if key not in CITIES:
        return {"error": f"невідоме місто: {city}"}
    lat, lon = CITIES[key]
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=20, params={
        "latitude": lat, "longitude": lon, "timezone": "Europe/Kyiv", "start_date": date, "end_date": date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"})
    r.raise_for_status()
    d = r.json()["daily"]
    return {"city": city, "date": date, "t_max_c": d["temperature_2m_max"][0], "t_min_c": d["temperature_2m_min"][0],
            "precipitation_mm": d["precipitation_sum"][0], "wind_max_kmh": d["wind_speed_10m_max"][0]}


TOOLS = {"get_current_date": get_current_date, "get_weather": get_weather}


def run(question: str):
    config = types.GenerateContentConfig(
        tools=list(TOOLS.values()),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        system_instruction="Ти — помічник з погоди. Ніколи не вгадуй дату: дізнавайся її інструментом.")
    chat = client.chats.create(model=MODEL, config=config)
    print("=" * 70, "\nПитання:", question)
    resp = chat.send_message(question)
    while resp.function_calls:
        parts = []
        for fc in resp.function_calls:
            result = TOOLS[fc.name](**dict(fc.args))
            print(f"  -> {fc.name}({dict(fc.args)}) = {result}")
            parts.append(types.Part.from_function_response(name=fc.name, response=result))
        resp = chat.send_message(parts)
    print("Відповідь агента:", resp.text.strip())


if __name__ == "__main__":
    run("Яка погода буде завтра в Тернополі? Чи потрібна парасолька?")
    run("Порівняй максимальну температуру сьогодні у Львові та Києві.")
