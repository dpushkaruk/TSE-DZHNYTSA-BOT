from datetime import date

import pandas as pd
import pytest

from tests.fake_raw import make_raw
from tse_dzhynsa_bot import rule
from tse_dzhynsa_bot.features import (
    FEATURE_COLUMNS,
    add_channel_ad_rate,
    build_features,
    clean,
    read_raw_range,
    validate_raw,
)


@pytest.fixture
def raw(tmp_path):
    make_raw(tmp_path, ["a", "b"], date(2026, 1, 1), date(2026, 1, 5))
    return read_raw_range(tmp_path, date(2026, 1, 1), date(2026, 1, 5))


def test_validate_rejects_missing_text(raw):
    with pytest.raises(ValueError, match="missing columns"):
        validate_raw(raw.drop(columns=["text"]))


def test_clean_strips_mark_and_dedups(raw):
    doubled = pd.concat([raw, raw])
    c = clean(doubled)
    assert len(c) == len(raw)
    assert not c["text_clean"].str.contains("#реклама").any()
    assert c["label_ad"].sum() > 0


def test_features_have_all_columns(raw):
    f = build_features(clean(raw))
    assert set(FEATURE_COLUMNS) <= set(f.columns)
    assert (f["feature_version"] == "v1").all()
    assert rule.predict(f).isin([0, 1]).all()


def test_channel_ad_rate_is_point_in_time():
    df = pd.DataFrame(
        {
            "channel": ["a"] * 3,
            "day": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)],
            "label_ad": [1, 0, 0],
        }
    )
    rate = add_channel_ad_rate(df)
    # day 1 sees no history; day 2 sees only day 1 (100 % ads); day 3 sees days 1-2
    assert rate.tolist() == [0.0, 1.0, 0.5]


def test_channel_ad_rate_handles_missing_days():
    df = pd.DataFrame(
        {
            "channel": ["a", "a"],
            "day": [date(2026, 1, 1), date(2026, 1, 4)],
            "label_ad": [1, 0],
        }
    )
    assert add_channel_ad_rate(df).tolist() == [0.0, 1.0]
