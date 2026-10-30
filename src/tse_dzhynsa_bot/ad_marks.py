"""Detect and strip the channel's own advertising mark (design.md 5.2)."""

import re
from functools import lru_cache
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "ad_marks.yaml"


@lru_cache
def load_patterns(path: Path = DEFAULT_CONFIG) -> tuple[re.Pattern, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))["patterns"]
    return tuple(re.compile(p, re.IGNORECASE | re.MULTILINE) for p in raw)


def has_mark(text: str, patterns: tuple[re.Pattern, ...] | None = None) -> bool:
    patterns = patterns or load_patterns()
    return any(p.search(text) for p in patterns)


def strip_mark(text: str, patterns: tuple[re.Pattern, ...] | None = None) -> str:
    patterns = patterns or load_patterns()
    for p in patterns:
        text = p.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()
