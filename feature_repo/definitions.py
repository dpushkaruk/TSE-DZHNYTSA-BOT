"""Feast definitions, registered by `feast apply`:

    uv run --env-file .env feast -c feature_repo apply

Both views read the warehouse table `features`, which the DAG fills (build_features).
- post_features: one row per post (channel + post_id), event time = post time.
- channel_stats: one row per channel and Kyiv day, event time = start of that day;
  channel_ad_rate_30d already looks only at earlier days, so it is known at that moment.
Training joins both point-in-time; serving computes the post features from the new
post and reads only channel_stats from the online store.
"""

from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, ValueType
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, String

from tse_dzhynsa_bot.features import (
    CHANNEL_FEATURE_COLUMNS,
    FEATURE_VERSION,
    POST_FEATURE_COLUMNS,
)

channel = Entity(name="channel", join_keys=["channel"], value_type=ValueType.STRING)
post = Entity(name="post", join_keys=["post_id"], value_type=ValueType.INT64)

post_source = PostgreSQLSource(
    name="features_post",
    query=(
        "SELECT channel, id AS post_id, ts AS event_ts, text_clean, "
        + ", ".join(POST_FEATURE_COLUMNS)
        + f" FROM features WHERE feature_version = '{FEATURE_VERSION}'"
    ),
    timestamp_field="event_ts",
)

channel_source = PostgreSQLSource(
    name="features_channel_day",
    query=(
        "SELECT channel, (day::timestamp AT TIME ZONE 'Europe/Kyiv') AS event_ts, "
        + ", ".join(f"max({c}) AS {c}" for c in CHANNEL_FEATURE_COLUMNS)
        + f" FROM features WHERE feature_version = '{FEATURE_VERSION}' GROUP BY channel, day"
    ),
    timestamp_field="event_ts",
)

post_features = FeatureView(
    name="post_features",
    entities=[channel, post],
    ttl=timedelta(days=1),
    schema=[Field(name="text_clean", dtype=String)]
    + [Field(name=c, dtype=Float64) for c in POST_FEATURE_COLUMNS],
    source=post_source,
    online=False,  # computed from the new post at serving time
    tags={"feature_version": FEATURE_VERSION},
)

channel_stats = FeatureView(
    name="channel_stats",
    entities=[channel],
    ttl=timedelta(days=7),  # a channel silent for a week falls back to 0
    schema=[Field(name=c, dtype=Float64) for c in CHANNEL_FEATURE_COLUMNS],
    source=channel_source,
    online=True,
    tags={"feature_version": FEATURE_VERSION},
)

# what the model consumes: train.py --source feast asks for this service
dzhynsa_v1 = FeatureService(
    name=f"dzhynsa_{FEATURE_VERSION}", features=[post_features, channel_stats]
)
