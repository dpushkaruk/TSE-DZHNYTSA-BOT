from datetime import date

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from tests.fake_raw import make_raw
from tse_dzhynsa_bot import api
from tse_dzhynsa_bot.predict import ChampionLoader, score
from tse_dzhynsa_bot.train import PARAMS, load_dataset, make_model

NAME = "dzhynsa-test"


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """Local MLflow (sqlite) with one registered model at @champion, threshold 0.4."""
    tmp = tmp_path_factory.mktemp("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp / 'mlflow.db'}")
    make_raw(tmp / "raw", ["a", "b"], date(2026, 8, 25), date(2026, 9, 5), per_day=12)
    feats = load_dataset(tmp / "raw")
    model = make_model(PARAMS | {"min_df": 1}).fit(feats, feats["label_ad"])
    client = MlflowClient()
    for thr in (0.4, 0.6):  # two versions, the alias moves to the second later
        with mlflow.start_run():
            info = mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")
        mv = mlflow.register_model(info.model_uri, NAME)
        client.set_model_version_tag(NAME, mv.version, "threshold", str(thr))
    client.set_registered_model_alias(NAME, "champion", "1")
    yield feats
    mlflow.set_tracking_uri(None)


def test_verdicts(registry):
    champ = ChampionLoader(NAME).get()
    assert (champ.version, champ.threshold) == ("1", 0.4)
    out = score(champ, registry)
    marked = registry["label_ad"] == 1
    assert (out.loc[marked, "verdict"] == "ad").all()
    unmarked = out[~marked]
    assert (unmarked["verdict"] == "dzhynsa").eq(unmarked["p_ad"] >= 0.4).all()
    assert set(out["verdict"]) <= {"ad", "dzhynsa", "regular"}
    assert (out["model_version"] == f"{NAME}/1").all()


def test_new_champion_is_picked_up(registry):
    loader = ChampionLoader(NAME, check_every=0)
    assert loader.get().version == "1"
    MlflowClient().set_registered_model_alias(NAME, "champion", "2")
    try:
        assert (loader.get().version, loader.get().threshold) == ("2", 0.6)
    finally:
        MlflowClient().set_registered_model_alias(NAME, "champion", "1")


def test_score_endpoint(registry, monkeypatch):
    monkeypatch.setattr(api, "loader", ChampionLoader(NAME))
    monkeypatch.setattr(api, "channel_rate", lambda channel, day: 0.1)
    client = TestClient(api.app)
    assert client.get("/health").json()["model"] == f"{NAME}/1"

    ad = client.post("/score", json={"text": "Знижка 30% на кросівки\n#реклама", "channel": "a"})
    assert ad.status_code == 200
    assert ad.json()["verdict"] == "ad" and ad.json()["marked"]

    news = client.post("/score", json={"text": "Вибухи у Києві, працює ППО", "channel": "a"})
    body = news.json()
    assert body["verdict"] in {"regular", "dzhynsa"} and not body["marked"]
    assert body["channel_ad_rate_30d"] == 0.1
    assert client.post("/score", json={"text": "   "}).status_code == 422
