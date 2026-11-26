"""Навчальний модуль: обробка оцінок студентів (коректна реалізація)."""


def average(scores: list[float]) -> float:
    """Середнє арифметичне. Порожній список -> ValueError."""
    if not scores:
        raise ValueError("список оцінок порожній")
    return sum(scores) / len(scores)


def letter_grade(score: float) -> str:
    """Оцінка ECTS за 100-бальною шкалою ТНПУ: A 90-100, B 82-89, C 74-81, D 64-73, E 60-63, FX 35-59, F 0-34."""
    if not 0 <= score <= 100:
        raise ValueError("бал має бути в межах 0..100")
    for low, letter in ((90, "A"), (82, "B"), (74, "C"), (64, "D"), (60, "E"), (35, "FX")):
        if score >= low:
            return letter
    return "F"


def parse_scores(line: str) -> list[int]:
    """Розбирає рядок '95, 88;72' у список цілих балів; роздільники кома, крапка з комою або пробіл."""
    parts = [p for p in line.replace(";", ",").replace(" ", ",").split(",") if p]
    result = []
    for p in parts:
        if not p.lstrip("-").isdigit():
            raise ValueError(f"некоректний бал: {p!r}")
        result.append(int(p))
    return result
