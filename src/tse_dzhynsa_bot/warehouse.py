"""PostgreSQL warehouse: posts_clean and features. Every write replaces one day
(delete, then insert, in one transaction), so rerunning a day never duplicates rows."""

import os
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from tse_dzhynsa_bot.features import FEATURE_COLUMNS, FEATURE_VERSION

DDL = f"""
CREATE TABLE IF NOT EXISTS posts_clean (
    channel TEXT NOT NULL, id BIGINT NOT NULL, day DATE NOT NULL, ts TIMESTAMPTZ NOT NULL,
    text_clean TEXT NOT NULL, label_ad SMALLINT NOT NULL, links JSONB, forward_from BIGINT,
    views BIGINT, forwards BIGINT, PRIMARY KEY (channel, id)
);
CREATE INDEX IF NOT EXISTS posts_clean_day ON posts_clean (day);
CREATE TABLE IF NOT EXISTS features (
    channel TEXT NOT NULL, id BIGINT NOT NULL, day DATE NOT NULL, ts TIMESTAMPTZ NOT NULL,
    feature_version TEXT NOT NULL, text_clean TEXT NOT NULL, link_domains TEXT,
    label_ad SMALLINT NOT NULL,
    {", ".join(f"{c} DOUBLE PRECISION" for c in FEATURE_COLUMNS)},
    PRIMARY KEY (channel, id, feature_version)
);
CREATE INDEX IF NOT EXISTS features_day ON features (day, feature_version);
CREATE TABLE IF NOT EXISTS predictions (
    channel TEXT NOT NULL, id BIGINT NOT NULL, day DATE NOT NULL, p_ad DOUBLE PRECISION NOT NULL,
    label_ad SMALLINT NOT NULL, verdict TEXT NOT NULL, model_version TEXT NOT NULL,
    threshold DOUBLE PRECISION NOT NULL, feature_version TEXT NOT NULL,
    scored_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (channel, id, model_version)
);
CREATE INDEX IF NOT EXISTS predictions_day ON predictions (day, verdict);
"""


LOCAL_URL = "postgresql+psycopg2://dzhynsa:dzhynsa@localhost:5432/warehouse"


def engine() -> Engine:
    return create_engine(os.environ.get("WAREHOUSE_URL", LOCAL_URL))


def init(eng: Engine) -> None:
    with eng.begin() as con:
        con.execute(text(DDL))


def replace_day(eng: Engine, table: str, day: date, df: pd.DataFrame, where: str = "") -> int:
    with eng.begin() as con:
        con.execute(text(f"DELETE FROM {table} WHERE day = :day {where}"), {"day": day})
        if len(df):
            df.to_sql(table, con, if_exists="append", index=False, method="multi", chunksize=500)
        return con.execute(
            text(f"SELECT count(*) FROM {table} WHERE day = :day {where}"), {"day": day}
        ).scalar_one()


def read_clean(eng: Engine, start: date, end: date) -> pd.DataFrame:
    q = text("SELECT * FROM posts_clean WHERE day BETWEEN :s AND :e")
    df = pd.read_sql(q, eng, params={"s": start, "e": end})
    df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_convert("Europe/Kyiv")
    df["links"] = df["links"].map(lambda v: v if isinstance(v, list) else [])
    return df


def read_features(eng: Engine, day: date, version: str = FEATURE_VERSION) -> pd.DataFrame:
    q = text("SELECT * FROM features WHERE day = :d AND feature_version = :v")
    return pd.read_sql(q, eng, params={"d": day, "v": version})


def channel_ad_rate(eng: Engine, channel: str, day: date, window_days: int = 30) -> float:
    """Same as features.add_channel_ad_rate for one channel: marked-ad share over the
    `window_days` days before `day` (the day itself excluded)."""
    q = text(
        "SELECT coalesce(avg(label_ad), 0) FROM posts_clean "
        "WHERE channel = :c AND day >= CAST(:d AS date) - :w AND day < :d"
    )
    with eng.connect() as con:
        return float(con.execute(q, {"c": channel, "d": day, "w": window_days}).scalar_one())


def count(eng: Engine, sql: str, **params) -> int:
    with eng.connect() as con:
        return con.execute(text(sql), params).scalar_one()
