"""Hand rule baseline: the trained model has to beat this (lab 2, step 5)."""

import re

import pandas as pd

AD_WORDS = re.compile(
    r"знижк|акці[яї]|замовля|переходь\w* за посиланням|за посиланням у біо|"
    r"промокод|розіграш|купуй|доставк|ціна|грн\b|бот[іа]?\b|@\w+bot\b",
    re.IGNORECASE,
)


def predict(features: pd.DataFrame) -> pd.Series:
    words = features["text_clean"].str.contains(AD_WORDS)
    signals = features["has_promo_code"] + features["has_phone"] + features["has_invite_link"]
    return ((words & (features["n_links"] > 0)) | (signals > 0)).astype(int)
