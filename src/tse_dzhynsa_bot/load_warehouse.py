"""Fill the warehouse (posts_clean, features) for a range of days from data/raw in one go,
so Feast has the full history without 150 Airflow runs. Same code as the DAG
(clean, build_features); each day is replaced, so a rerun gives the same rows.

    uv run --env-file .env python -m tse_dzhynsa_bot.load_warehouse \
        --start 2026-06-01 --end 2026-10-27
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

from tse_dzhynsa_bot import warehouse as wh
from tse_dzhynsa_bot.features import FEATURE_VERSION, build_features, clean, read_raw_range

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--start", type=date.fromisoformat, default=date(2026, 6, 1))
    ap.add_argument("--end", type=date.fromisoformat, default=date(2026, 10, 27))
    args = ap.parse_args()

    # 30 extra days in front so channel_ad_rate_30d of the first day is point-in-time too
    raw = read_raw_range(args.raw, args.start - timedelta(days=30), args.end)
    posts = clean(raw)
    feats = build_features(posts)
    posts["links"] = posts["links"].map(lambda v: json.dumps(list(v)))
    eng = wh.engine()
    wh.init(eng)
    total = 0
    day = args.start
    while day <= args.end:
        wh.replace_day(eng, "posts_clean", day, posts[posts["day"] == day])
        total += wh.replace_day(
            eng,
            "features",
            day,
            feats[feats["day"] == day],
            f"AND feature_version = '{FEATURE_VERSION}'",
        )
        day += timedelta(days=1)
    print(f"warehouse {args.start}..{args.end}: {total} feature rows, version {FEATURE_VERSION}")


if __name__ == "__main__":
    main()
