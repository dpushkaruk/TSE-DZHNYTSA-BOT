"""Feast definitions match the feature code (no database needed)."""

import importlib.util
from pathlib import Path

from tse_dzhynsa_bot.features import CHANNEL_FEATURE_COLUMNS, FEATURE_COLUMNS, POST_FEATURE_COLUMNS

REPO = Path(__file__).resolve().parents[1] / "feature_repo"


def load_definitions():
    spec = importlib.util.spec_from_file_location("feast_definitions", REPO / "definitions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_views_cover_all_model_features():
    d = load_definitions()
    post = {f.name for f in d.post_features.schema}
    channel = {f.name for f in d.channel_stats.schema}
    assert post == {"text_clean", *POST_FEATURE_COLUMNS}
    assert channel == set(CHANNEL_FEATURE_COLUMNS)
    assert sorted(POST_FEATURE_COLUMNS + CHANNEL_FEATURE_COLUMNS) == sorted(FEATURE_COLUMNS)


def test_only_channel_stats_is_served_online():
    d = load_definitions()
    assert d.channel_stats.online and not d.post_features.online
    assert d.dzhynsa_v1.name == "dzhynsa_v1"
