"""Raw posts -> clean rows -> feature rows. The DAG and training both call these
functions, so a feature is computed by the same code at training and serving time."""

import gzip
import json
import re
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from tse_dzhynsa_bot.ad_marks import has_mark, strip_mark

FEATURE_VERSION = "v1"
REQUIRED_COLUMNS = {"channel": str, "id": int, "date": str, "text": str}
FEATURE_COLUMNS = [
    "n_links",
    "has_invite_link",
    "has_promo_code",
    "has_phone",
    "text_len",
    "is_forward",
    "hour_of_day",
    "channel_ad_rate_30d",
]

PROMO_RE = re.compile(r"промокод|promo\s?code|знижк\w*\s+\d+\s?%", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:\+?38)?\s?\(?0\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")
INVITE_RE = re.compile(r"t\.me/(?:\+|joinchat/)", re.IGNORECASE)


def read_raw_day(raw_dir: Path, day: date) -> pd.DataFrame:
    rows = []
    for f in sorted(Path(raw_dir).glob(f"*/{day.isoformat()}.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh)
    return pd.DataFrame(rows)


def read_raw_range(raw_dir: Path, start: date, end: date) -> pd.DataFrame:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    frames = [read_raw_day(raw_dir, d) for d in days]
    return pd.concat([f for f in frames if len(f)], ignore_index=True)


def validate_raw(df: pd.DataFrame) -> None:
    """Schema check: fails the batch before anything is written."""
    if df.empty:
        raise ValueError("batch is empty")
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if df["id"].isna().any() or df["channel"].isna().any():
        raise ValueError("null channel or id")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df = df[df["text"].str.strip() != ""]
    df = df.drop_duplicates(subset=["channel", "id"], keep="last")
    df["ts"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/Kyiv")
    df["day"] = df["ts"].dt.date
    df["label_ad"] = df["text"].map(has_mark).astype(int)
    df["text_clean"] = df["text"].map(strip_mark)
    for col in ("links", "forward_from", "views", "forwards"):
        if col not in df.columns:
            df[col] = None
    df["links"] = df["links"].map(lambda v: v if isinstance(v, list) else [])
    cols = ["channel", "id", "ts", "day", "text_clean", "label_ad", "links", "forward_from"]
    return df[cols + ["views", "forwards"]].reset_index(drop=True)


def _domains(links: list[str]) -> str:
    return ",".join(sorted({urlparse(u).netloc.lower() for u in links if u}))


def add_channel_ad_rate(clean_df: pd.DataFrame, window_days: int = 30) -> pd.Series:
    """Share of marked ads per channel over the previous `window_days` days,
    excluding the post's own day (point-in-time: no look at the future)."""
    daily = clean_df.groupby(["channel", "day"])["label_ad"].agg(["sum", "count"]).reset_index()
    daily["day"] = pd.to_datetime(daily["day"])
    out = []
    for channel, g in daily.groupby("channel"):
        g = g.set_index("day")[["sum", "count"]].asfreq("D", fill_value=0)
        roll = g.rolling(f"{window_days}D").sum().shift(1, freq="D")
        g["rate"] = (roll["sum"] / roll["count"]).reindex(g.index)
        g["channel"] = channel
        out.append(g.reset_index()[["channel", "day", "rate"]])
    rates = pd.concat(out)
    rates["day"] = rates["day"].dt.date
    merged = clean_df[["channel", "day"]].merge(rates, on=["channel", "day"], how="left")
    return merged["rate"].fillna(0.0).astype(float)


def build_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """One row per post. Text with the mark removed is the main feature;
    the rest are small numeric signals known at serving time."""
    df = clean_df.copy()
    text = df["text_clean"]
    df["link_domains"] = df["links"].map(_domains)
    df["n_links"] = df["links"].map(len).astype(int) + text.str.count(r"https?://")
    df["has_invite_link"] = (
        df["links"].map(lambda ls: any(INVITE_RE.search(u) for u in ls))
        | text.str.contains(INVITE_RE)
    ).astype(int)
    df["has_promo_code"] = text.str.contains(PROMO_RE).astype(int)
    df["has_phone"] = text.str.contains(PHONE_RE).astype(int)
    df["text_len"] = text.str.len().astype(int)
    df["is_forward"] = df["forward_from"].notna().astype(int)
    df["hour_of_day"] = df["ts"].dt.hour.astype(int)
    df["channel_ad_rate_30d"] = add_channel_ad_rate(df)
    df["feature_version"] = FEATURE_VERSION
    keep = ["channel", "id", "day", "ts", "text_clean", "link_domains", "label_ad"]
    return df[keep + FEATURE_COLUMNS + ["feature_version"]]
