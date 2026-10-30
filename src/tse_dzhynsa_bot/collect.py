"""Collect posts from the channels in configs/channels.yaml with Telethon.

One file per channel and Kyiv day: data/raw/<channel>/<YYYY-MM-DD>.jsonl.gz.
A day is always rewritten in full and sorted by id, so running the same
interval twice gives the same files (idempotent).

    uv run python -m tse_dzhynsa_bot.collect --since 2026-06-01 --until 2026-10-27
"""

import argparse
import asyncio
import gzip
import json
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv
from telethon import TelegramClient
from telethon.tl.types import MessageEntityTextUrl, MessageEntityUrl

KYIV = ZoneInfo("Europe/Kyiv")
ROOT = Path(__file__).resolve().parents[2]


def to_record(channel: str, m) -> dict:
    text = m.message or ""
    links = []
    for ent in m.entities or []:
        if isinstance(ent, MessageEntityTextUrl):
            links.append(ent.url)
        elif isinstance(ent, MessageEntityUrl):
            links.append(text[ent.offset : ent.offset + ent.length])
    fwd = m.fwd_from
    return {
        "channel": channel,
        "id": m.id,
        "date": m.date.astimezone(KYIV).isoformat(),
        "text": text,
        "views": m.views,
        "forwards": m.forwards,
        "forward_from": getattr(getattr(fwd, "from_id", None), "channel_id", None) if fwd else None,
        "links": links,
        "edit_date": m.edit_date.astimezone(KYIV).isoformat() if m.edit_date else None,
        "grouped_id": m.grouped_id,
    }


def write_day(out_dir: Path, channel: str, day: date, rows: list[dict]) -> Path:
    path = out_dir / channel / f"{day.isoformat()}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: r["id"])
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    # mtime=0 keeps the gzip bytes identical between runs, so DVC sees no change
    with open(path, "wb") as f, gzip.GzipFile(fileobj=f, mode="wb", mtime=0) as gz:
        gz.write(body.encode("utf-8"))
    return path


async def collect(channels: list[str], since: date, until: date, out_dir: Path) -> dict:
    start = datetime.combine(since, datetime.min.time(), KYIV)
    end = datetime.combine(until + timedelta(days=1), datetime.min.time(), KYIV)
    counts = {}
    client = TelegramClient(
        str(ROOT / os.environ.get("TG_SESSION", "collector")),
        int(os.environ["TG_API_ID"]),
        os.environ["TG_API_HASH"],
        flood_sleep_threshold=120,
    )
    async with client:
        for channel in channels:
            days = defaultdict(list)
            # newest first, starting just before `end`; stop once we pass `start`
            async for m in client.iter_messages(channel, offset_date=end):
                if m.date < start:
                    break
                if m.action is not None:
                    continue
                days[m.date.astimezone(KYIV).date()].append(to_record(channel, m))
            d = since
            while d <= until:
                write_day(out_dir, channel, d, days.get(d, []))
                d += timedelta(days=1)
            counts[channel] = sum(len(v) for v in days.values())
            print(f"{channel}: {counts[channel]} posts")
    return counts


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser()
    yesterday = datetime.now(KYIV).date() - timedelta(days=1)
    ap.add_argument("--since", type=date.fromisoformat, default=yesterday - timedelta(days=2))
    ap.add_argument("--until", type=date.fromisoformat, default=yesterday)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--channels", nargs="*")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs" / "channels.yaml").read_text(encoding="utf-8"))
    channels = args.channels or cfg["channels"]
    asyncio.run(collect(channels, args.since, args.until, args.out))


if __name__ == "__main__":
    main()
