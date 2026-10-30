"""ingest_daily: one Kyiv day of posts, from the raw files to the feature table.

The batch is the run's logical date (= data interval start), never the current time.
Raw files come from data/raw (pulled with `dvc pull`, filled by the Telethon
collector), so a reviewer can run any day without Telegram keys. Every load replaces
its day: safe to rerun.
"""

from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.sdk import dag, get_current_context, task
from airflow.timetables.interval import CronDataIntervalTimetable

PROJECT = Path("/opt/project")
STAGING = Path("/opt/airflow/staging")  # inside the container, not the repo


def batch_day():
    # With CronDataIntervalTimetable the logical date is the start of the data
    # interval for scheduled runs; manual runs get only a logical date, so key on it.
    return get_current_context()["logical_date"].in_timezone("Europe/Kyiv").date()


def staging(day, name):
    p = STAGING / day.isoformat()
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{name}.parquet"


@dag(
    schedule=CronDataIntervalTimetable("@daily", timezone="Europe/Kyiv"),
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Kyiv"),
    catchup=False,
    max_active_runs=1,
    params={"raw_dir": str(PROJECT / "data" / "raw")},
    default_args={"retries": 0},
    tags=["lab2"],
)
def ingest_daily():
    @task
    def extract() -> int:
        from tse_dzhynsa_bot.features import read_raw_day

        day = batch_day()
        df = read_raw_day(Path(get_current_context()["params"]["raw_dir"]), day)
        df.to_parquet(staging(day, "raw"), index=False)
        print(f"extract {day}: {len(df)} raw posts")
        return len(df)

    @task
    def validate() -> None:
        import pandas as pd

        from tse_dzhynsa_bot.features import validate_raw

        validate_raw(pd.read_parquet(staging(batch_day(), "raw")))

    @task
    def transform() -> int:
        import pandas as pd

        from tse_dzhynsa_bot.features import clean

        day = batch_day()
        df = clean(pd.read_parquet(staging(day, "raw")))
        df = df[df["day"] == day]
        df.to_parquet(staging(day, "clean"), index=False)
        return len(df)

    @task
    def load() -> int:
        import json

        import pandas as pd

        from tse_dzhynsa_bot import warehouse as wh

        day = batch_day()
        df = pd.read_parquet(staging(day, "clean"))
        df["links"] = df["links"].map(lambda v: json.dumps(list(v)))
        eng = wh.engine()
        wh.init(eng)
        n = wh.replace_day(eng, "posts_clean", day, df)
        print(f"load {day}: posts_clean has {n} rows for the day")
        return n

    @task
    def build_features() -> int:
        from tse_dzhynsa_bot import warehouse as wh
        from tse_dzhynsa_bot.features import FEATURE_VERSION
        from tse_dzhynsa_bot.features import build_features as build

        day = batch_day()
        eng = wh.engine()
        # 30 days of history so channel_ad_rate_30d is point-in-time
        hist = wh.read_clean(eng, day - timedelta(days=30), day)
        feats = build(hist)
        feats = feats[feats["day"] == day]
        n = wh.replace_day(
            eng, "features", day, feats, f"AND feature_version = '{FEATURE_VERSION}'"
        )
        print(f"build_features {day}: {n} rows, version {FEATURE_VERSION}")
        return n

    @task
    def quality_check(raw_rows: int, clean_rows: int, loaded: int, feature_rows: int) -> None:
        import yaml

        from tse_dzhynsa_bot import warehouse as wh

        cfg = yaml.safe_load((PROJECT / "configs" / "quality.yaml").read_text())
        day = batch_day()
        eng = wh.engine()
        channels = wh.count(
            eng, "SELECT count(DISTINCT channel) FROM posts_clean WHERE day=:d", d=day
        )
        ads = wh.count(eng, "SELECT coalesce(sum(label_ad),0) FROM posts_clean WHERE day=:d", d=day)
        errors = []
        if clean_rows < cfg["min_rows"]:
            errors.append(f"{clean_rows} clean rows < {cfg['min_rows']}")
        if channels < cfg["min_channels"]:
            errors.append(f"{channels} channels < {cfg['min_channels']}")
        empty_share = 1 - clean_rows / raw_rows if raw_rows else 1
        if empty_share > cfg["max_empty_text_share"]:
            errors.append(f"empty-text share {empty_share:.2f} > {cfg['max_empty_text_share']}")
        if clean_rows and ads / clean_rows > cfg["max_ad_share"]:
            errors.append(f"ad share {ads / clean_rows:.2f} > {cfg['max_ad_share']}")
        if not (clean_rows == loaded == feature_rows):
            errors.append(
                f"row counts differ: clean {clean_rows}, loaded {loaded}, features {feature_rows}"
            )
        print(
            f"quality_check {day}: raw={raw_rows} clean={clean_rows} channels={channels} ads={ads}"
        )
        if errors:
            raise ValueError("quality gate failed: " + "; ".join(errors))

    raw = extract()
    v = validate()
    t = transform()
    ld = load()
    f = build_features()
    raw >> v >> t >> ld >> f
    quality_check(raw, t, ld, f)


ingest_daily()
