"""Inference with the registered champion: one day of posts into the `predictions` table,
or single posts for the API (api.py).

    uv run --env-file .env python -m tse_dzhynsa_bot.predict --day 2026-09-10

The model is loaded by alias (`models:/dzhynsa-tfidf@champion`), so moving the alias to a
new version is enough to deploy it; the decision threshold is the version's `threshold` tag.
MODEL_NAME / MODEL_ALIAS (env) switch to another registered model, e.g. dzhynsa-xlmr:
an sklearn model is scored with predict_proba, any other pyfunc must return `prob_ad`.
Verdict per post (design.md 5.2):
  ad       - the channel marked it itself (rule, no model needed)
  dzhynsa  - no mark, but P(ad) >= threshold: suspected hidden ad
  regular  - everything else
"""

import argparse
import os
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from tse_dzhynsa_bot.features import FEATURE_COLUMNS, FEATURE_VERSION
from tse_dzhynsa_bot.train import MODEL_NAME

NAME = os.environ.get("MODEL_NAME", MODEL_NAME)
ALIAS = os.environ.get("MODEL_ALIAS", "champion")
PREDICTION_COLUMNS = [
    "channel",
    "id",
    "day",
    "p_ad",
    "label_ad",
    "verdict",
    "model_version",
    "threshold",
    "feature_version",
    "scored_at",
]


@dataclass
class Champion:
    model: object
    version: str
    threshold: float
    run_id: str
    name: str = MODEL_NAME

    def proba(self, feats: pd.DataFrame):
        x = feats[["text_clean", *FEATURE_COLUMNS]]
        if hasattr(self.model, "predict_proba"):  # sklearn flavor
            return self.model.predict_proba(x)[:, 1]
        return pd.DataFrame(self.model.predict(x))["prob_ad"].to_numpy()  # pyfunc


def load_model(uri: str):
    try:
        return mlflow.sklearn.load_model(uri)
    except MlflowException:  # no sklearn flavor: generic pyfunc
        return mlflow.pyfunc.load_model(uri)


class ChampionLoader:
    """Keeps the champion in memory and reloads it when the alias moves to another version.
    The alias is checked at most every `check_every` seconds (one cheap registry call)."""

    def __init__(self, name: str = NAME, alias: str = ALIAS, check_every: float = 60):
        self.name, self.alias, self.check_every = name, alias, check_every
        self._current: Champion | None = None
        self._checked = 0.0

    def get(self) -> Champion:
        if self._current is None or time.monotonic() - self._checked > self.check_every:
            mv = MlflowClient().get_model_version_by_alias(self.name, self.alias)
            self._checked = time.monotonic()
            if self._current is None or self._current.version != str(mv.version):
                model = load_model(f"models:/{self.name}@{self.alias}")
                thr = float(mv.tags.get("threshold", 0.5))
                self._current = Champion(model, str(mv.version), thr, mv.run_id, self.name)
                print(f"loaded {self.name} v{mv.version} (@{self.alias}), threshold {thr}")
        return self._current


def score(champ: Champion, feats: pd.DataFrame) -> pd.DataFrame:
    """feats: rows of the `features` table (text_clean + FEATURE_COLUMNS + label_ad)."""
    out = feats[["channel", "id", "day", "label_ad"]].copy()
    out["p_ad"] = champ.proba(feats)
    marked = out["label_ad"].astype(int) == 1
    out["verdict"] = "regular"
    out.loc[~marked & (out["p_ad"] >= champ.threshold), "verdict"] = "dzhynsa"
    out.loc[marked, "verdict"] = "ad"
    out["model_version"] = f"{champ.name}/{champ.version}"
    out["threshold"] = champ.threshold
    out["feature_version"] = FEATURE_VERSION
    out["scored_at"] = datetime.now(UTC)
    return out[PREDICTION_COLUMNS]


def predict_day(eng, day: date, champ: Champion) -> dict:
    """Score every post of one day from the features table; replaces that day's
    predictions of this model version, so a rerun never duplicates rows."""
    from tse_dzhynsa_bot import warehouse as wh

    wh.init(eng)
    feats = wh.read_features(eng, day)
    preds = score(champ, feats) if len(feats) else pd.DataFrame(columns=PREDICTION_COLUMNS)
    n = wh.replace_day(
        eng, "predictions", day, preds, f"AND model_version = '{champ.name}/{int(champ.version)}'"
    )
    counts = preds["verdict"].value_counts().to_dict()
    return {"day": day.isoformat(), "model": f"{champ.name}/{champ.version}", "rows": n} | {
        v: int(counts.get(v, 0)) for v in ("ad", "dzhynsa", "regular")
    }


def main() -> None:
    from tse_dzhynsa_bot import warehouse as wh

    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=date.fromisoformat, required=True)
    args = ap.parse_args()
    print(predict_day(wh.engine(), args.day, ChampionLoader().get()))


if __name__ == "__main__":
    main()
