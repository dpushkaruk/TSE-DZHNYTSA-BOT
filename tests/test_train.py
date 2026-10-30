from datetime import date

import numpy as np

from tests.fake_raw import make_raw
from tse_dzhynsa_bot.train import PARAMS, load_dataset, make_model, pick_threshold, split


def test_split_is_by_time_and_model_trains(tmp_path, monkeypatch):
    make_raw(tmp_path, ["a", "b", "c"], date(2026, 8, 20), date(2026, 10, 5), per_day=12)
    parts = split(load_dataset(tmp_path))
    assert parts["train"]["day"].max() < parts["val"]["day"].min()
    assert parts["val"]["day"].max() < parts["test"]["day"].min()
    model = make_model(PARAMS | {"min_df": 1}).fit(parts["train"], parts["train"]["label_ad"])
    prob = model.predict_proba(parts["test"])[:, 1]
    assert prob.shape == (len(parts["test"]),)


def test_threshold_respects_precision():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.6, 0.7, 0.9])
    t = pick_threshold(y, p, 0.99)
    assert 0.6 < t <= 0.7
