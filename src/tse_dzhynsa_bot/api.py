"""Scoring API (FastAPI). The Airflow DAG calls /predict-day after each daily batch;
the Lab 5 agent calls /score for single posts.

    uv run --env-file .env uvicorn tse_dzhynsa_bot.api:app --port 8000
    docker compose up -d api            # http://localhost:8000/docs

The champion is reloaded when its alias moves (checked at most once a minute).
"""

from datetime import UTC, date, datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tse_dzhynsa_bot import warehouse as wh
from tse_dzhynsa_bot.features import build_features, clean
from tse_dzhynsa_bot.predict import ChampionLoader, predict_day, score

app = FastAPI(title="TSE_DZHYNSA_BOT scoring")
loader = ChampionLoader()
_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = wh.engine()
    return _engine


class Post(BaseModel):
    text: str = Field(description="post text as published, with the ad mark if there is one")
    channel: str = ""
    id: int = 0
    ts: datetime | None = None
    links: list[str] = []
    forward_from: int | None = None


class Scored(BaseModel):
    verdict: str  # ad | dzhynsa | regular
    p_ad: float
    marked: bool
    threshold: float
    model: str
    channel_ad_rate_30d: float


def channel_rate(channel: str, day: date) -> float:
    if not channel:
        return 0.0
    try:
        return wh.channel_ad_rate(engine(), channel, day)
    except Exception as e:  # warehouse down: score without the channel history
        print(f"channel_ad_rate unavailable ({e.__class__.__name__}); using 0")
        return 0.0


def post_features(post: Post) -> pd.DataFrame:
    """One post -> one feature row, with the same code as the DAG (features.py)."""
    ts = post.ts or datetime.now(UTC)
    raw = pd.DataFrame(
        [
            {
                "channel": post.channel,
                "id": post.id,
                "date": ts.isoformat(),
                "text": post.text,
                "links": post.links,
                "forward_from": post.forward_from,
            }
        ]
    )
    rows = clean(raw)
    if rows.empty:
        raise HTTPException(422, "empty text")
    try:  # channel_ad_rate_30d from the Feast online store (feature_repo/)
        from tse_dzhynsa_bot.feast_store import serving_frame
        from tse_dzhynsa_bot.features import post_level_features

        feats = serving_frame(post_level_features(rows))
        return feats.assign(**{c: rows[c].values for c in ("channel", "id", "day", "label_ad")})
    except ImportError:
        pass  # no Feast in this checkout
    except Exception as e:
        print(f"Feast online read failed ({e.__class__.__name__}: {e}); using the warehouse")
    feats = build_features(rows)
    feats["channel_ad_rate_30d"] = channel_rate(post.channel, rows["day"].iloc[0])
    return feats


@app.get("/health")
def health() -> dict:
    try:
        c = loader.get()
    except Exception as e:
        return {"status": "no_model", "error": str(e)[:300]}
    return {"status": "ok", "model": f"{c.name}/{c.version}", "threshold": c.threshold}


@app.post("/score", response_model=Scored)
def score_post(post: Post) -> Scored:
    feats = post_features(post)
    champ = loader.get()
    row = score(champ, feats).iloc[0]
    return Scored(
        verdict=row["verdict"],
        p_ad=float(row["p_ad"]),
        marked=bool(row["label_ad"]),
        threshold=champ.threshold,
        model=row["model_version"],
        channel_ad_rate_30d=float(feats["channel_ad_rate_30d"].iloc[0]),
    )


class DayRequest(BaseModel):
    day: date


@app.post("/predict-day")
def predict_one_day(req: DayRequest) -> dict:
    """Scores the day's rows of the features table into `predictions` (idempotent)."""
    return predict_day(engine(), req.day, loader.get())
