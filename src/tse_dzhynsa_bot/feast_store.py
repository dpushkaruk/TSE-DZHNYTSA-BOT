"""Feast access for training (offline, point-in-time) and serving (online).

    uv run --env-file .env feast -c feature_repo apply            # once, after a change
    uv run --env-file .env python -m tse_dzhynsa_bot.feast_store materialize \
        --start 2026-06-01 --end 2026-10-27                        # warehouse -> online store
    uv run --env-file .env python -m tse_dzhynsa_bot.feast_store online ssternenko k_dvizh
"""

import argparse
import os
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from tse_dzhynsa_bot.features import (
    CHANNEL_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    POST_FEATURE_COLUMNS,
    post_level_features,
)

REPO = Path(os.environ.get("FEAST_REPO", Path(__file__).resolve().parents[2] / "feature_repo"))
FEATURE_SERVICE = f"dzhynsa_{FEATURE_VERSION}"
KYIV = ZoneInfo("Europe/Kyiv")


def get_store():
    from feast import FeatureStore  # heavy import, only when Feast is used

    return FeatureStore(repo_path=str(REPO))


def kyiv_midnight(day: date) -> datetime:
    return datetime.combine(day, time(), tzinfo=KYIV)


def training_frame(start: date, end: date, store=None) -> pd.DataFrame:
    """Posts of [start, end] with labels, features joined point-in-time by Feast.
    Same columns as features.build_features, so train.py uses either source."""
    store = store or get_store()
    entity_sql = (
        "SELECT channel, id AS post_id, ts AS event_timestamp, label_ad FROM features "
        f"WHERE feature_version = '{FEATURE_VERSION}' "
        f"AND day BETWEEN '{start.isoformat()}' AND '{end.isoformat()}'"
    )
    df = store.get_historical_features(
        entity_df=entity_sql, features=store.get_feature_service(FEATURE_SERVICE)
    ).to_df()
    df = df.rename(columns={"post_id": "id"})
    df["ts"] = pd.to_datetime(df["event_timestamp"], utc=True).dt.tz_convert(KYIV)
    df["day"] = df["ts"].dt.date
    df["text_clean"] = df["text_clean"].fillna("")
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0.0).astype(float)
    df["label_ad"] = df["label_ad"].astype(int)
    cols = ["channel", "id", "day", "ts", "text_clean", "label_ad"] + FEATURE_COLUMNS
    return df[cols].sort_values(["ts", "channel", "id"]).reset_index(drop=True)


def online_channel_stats(channels: list[str], store=None) -> pd.DataFrame:
    store = store or get_store()
    return store.get_online_features(
        features=[f"channel_stats:{c}" for c in CHANNEL_FEATURE_COLUMNS],
        entity_rows=[{"channel": c} for c in channels],
    ).to_df()


def serving_frame(df: pd.DataFrame, store=None) -> pd.DataFrame:
    """New posts (clean-row columns) -> model input: post features computed here,
    channel_ad_rate_30d from the online store (0 for an unknown channel)."""
    if not set(POST_FEATURE_COLUMNS) <= set(df.columns):
        df = post_level_features(df)
    df = df.copy()
    online = online_channel_stats(df["channel"].tolist(), store)
    for c in CHANNEL_FEATURE_COLUMNS:
        df[c] = online[c].fillna(0.0).astype(float).to_numpy()
    return df


def materialize(start: date, end: date, store=None) -> None:
    """Copy channel_stats of Kyiv days [start, end] into the online store
    (the DAG calls it for its own day; rerunning a day writes the same values)."""
    store = store or get_store()
    store.materialize(
        start_date=kyiv_midnight(start),
        end_date=kyiv_midnight(end + timedelta(days=1)),
        feature_views=["channel_stats"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("materialize")
    m.add_argument("--start", type=date.fromisoformat, required=True)
    m.add_argument("--end", type=date.fromisoformat, required=True)
    o = sub.add_parser("online")
    o.add_argument("channels", nargs="+")
    args = ap.parse_args()
    if args.cmd == "materialize":
        materialize(args.start, args.end)
        print(f"materialized channel_stats {args.start}..{args.end}")
    else:
        print(online_channel_stats(args.channels).to_string(index=False))


if __name__ == "__main__":
    main()
