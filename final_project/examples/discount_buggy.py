"""Навчальний модуль: розрахунок вартості замовлення (містить приховані помилки).

Вимоги:
- товари з ціною від 100 (включно) отримують знижку 10 %;
- до суми додається податок 5 %;
- результат округлюється до 2 знаків;
- від'ємна ціна -> ValueError.
"""


def item_price(price: float) -> float:
    if price < 0:
        raise ValueError("ціна не може бути від'ємною")
    if price > 100:          # вимога: «від 100 включно»
        return price * 0.9
    return price


def order_total(prices: list[float]) -> float:
    subtotal = sum(item_price(p) for p in prices)
    return round(subtotal * 1.05, 1)   # вимога: 2 знаки після коми
