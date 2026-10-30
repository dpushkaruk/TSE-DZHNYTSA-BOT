from tse_dzhynsa_bot.ad_marks import has_mark, strip_mark


def test_marked_ad_detected_and_stripped():
    text = "Найкращі кредити тут 👉 bank.ua\n#реклама"
    assert has_mark(text)
    stripped = strip_mark(text)
    assert "реклама" not in stripped.lower()
    assert "bank.ua" in stripped


def test_regular_post_not_marked():
    assert not has_mark("Обстріл Харкова: є загиблі. Рекламації не приймаються.")


def test_partner_material():
    assert has_mark("Партнерський матеріал. Нова кавʼярня на Подолі")


def test_footer_ad_link_is_not_a_mark():
    assert not has_mark("Відбій тривоги\n\nПідписатися | Запропонувати новину | Реклама")
