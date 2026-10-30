"""Synthetic raw posts in the collector's format, for tests and local dry runs."""

import random
from datetime import date, datetime, timedelta
from pathlib import Path

from tse_dzhynsa_bot.collect import KYIV, write_day

NEWS = [
    "Вибухи у Києві, працює ППО",
    "Зеленський провів зустріч",
    "Ціни на пальне зросли",
    "У Львові відключення світла",
    "Новини фронту: ситуація на сході",
]
ADS = [
    "Знижка 30% на кросівки, замовляй за посиланням https://shop.ua/sale",
    "Промокод KYIV10 на доставку піци https://pizza.ua",
    "Найкращий курс англійської, переходь за посиланням https://t.me/+abc",
]


def make_raw(out: Path, channels: list[str], start: date, end: date, per_day: int = 10, seed=0):
    rnd = random.Random(seed)
    d = start
    while d <= end:
        for ch in channels:
            rows = []
            for i in range(per_day):
                ts = datetime.combine(d, datetime.min.time(), KYIV) + timedelta(minutes=60 * i + 5)
                is_ad = rnd.random() < 0.1
                text = rnd.choice(ADS if is_ad else NEWS) + f" #{d}-{i}"
                if is_ad and rnd.random() < 0.7:
                    text += "\n#реклама"
                rows.append(
                    {
                        "channel": ch,
                        "id": int(d.strftime("%j")) * 100 + i,
                        "date": ts.isoformat(),
                        "text": text,
                        "views": 1000,
                        "forwards": 3,
                        "forward_from": None,
                        "links": [],
                        "edit_date": None,
                        "grouped_id": None,
                    }
                )
            write_day(out, ch, d, rows)
        d += timedelta(days=1)
