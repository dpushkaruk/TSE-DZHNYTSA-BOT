"""Count marked ads per channel in data/raw (design.md 6.9: drop channels with ~0)."""

import gzip
import json
import sys
from pathlib import Path

from tse_dzhynsa_bot.ad_marks import has_mark

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    raw = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw"
    print(f"{'channel':<20}{'posts':>8}{'ads':>7}{'ad %':>7}")
    for ch_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        posts = ads = 0
        for f in ch_dir.glob("*.jsonl.gz"):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    posts += 1
                    ads += has_mark(json.loads(line)["text"])
        share = 100 * ads / posts if posts else 0
        print(f"{ch_dir.name:<20}{posts:>8}{ads:>7}{share:>6.1f}%")


if __name__ == "__main__":
    main()
