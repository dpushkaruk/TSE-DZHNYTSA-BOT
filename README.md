# TSE_DZHYNSA_BOT

Some posts in Ukrainian Telegram news channels are paid placements published without the advertising mark ("джинса").
This project collects new posts from 15 public Ukrainian channels once a day with Telethon.
Posts the channels mark as ads become free training labels: the mark is removed and a model learns what an ad looks like.
Every post is then classified as `ad`, `regular` or `dzhynsa` (unmarked, but the model says it reads like an ad), first with a TF-IDF baseline and later with a local LLM that sees similar posts.
The owner reviews the top-20 suspects weekly, and in Lab 5 a Telegram bot agent answers questions about which channels publish hidden ads.

Design document: [docs/design.md](docs/design.md).

## Maturity scorecard

| Property | Evidence a reviewer checks | Added in |
|---|---|---|
| Repository reproducible from a clean clone | CI is green; `make setup` and `make lint` succeed on a fresh clone | Lab 1 |
| Design document exists and is current | `docs/design.md` follows the template, every section filled or marked as an assumption | Lab 1 |
| Data versioned | `make pull` (DVC, MinIO remote `s3://dvc`) retrieves `data/raw`; checksum below | Lab 2 |
| Pipeline idempotent | the same day run twice gives the same row counts, pasted below | Lab 2 |
| Quality gate present | `make bad-batch` fails `quality_check`; screenshots in `docs/screenshots/` | Lab 2 |
| Experiments tracked | every run in MLflow experiment `dzhynsa-lab2` has params, metrics, `data_version`, `git_commit` | Lab 2 |
| Model registered with lineage | `dzhynsa-tfidf@champion` tagged with `data_version` and `git_commit`; `make reproduce` rebuilds it | Lab 2 |

## Quick start

```bash
make setup   # install the pinned environment with uv
make lint    # run ruff
uv run pre-commit install   # run ruff automatically before every commit
```

## Lab 2: data pipeline, versioning, baseline, MLflow

```bash
cp .env.example .env   # local defaults work as is
make up                # PostgreSQL 16, MinIO (pgsty/silo), MLflow :5000, Airflow :8080
make pull              # DVC: raw posts from MinIO into data/raw
make run DAY=2026-09-10      # one Kyiv day through ingest_daily
make count DAY=2026-09-10    # rows for that day
make rerun DAY=2026-09-10    # same day again, then make count again
make bad-batch         # quality gate fails on purpose
make train             # rule vs TF-IDF + LR, logged and registered as champion
make reproduce         # rebuild champion from its commit and data version
```

**Data.** 14 public channels (`configs/channels.yaml`), posts 2026-06-01..2026-10-27, collected by `make backfill` with the owner's Telegram account. One file per channel and Kyiv day: `data/raw/<channel>/<YYYY-MM-DD>.jsonl.gz`. DVC remote: S3 (MinIO, `s3://dvc`, endpoint `http://localhost:9000`), config in `.dvc/config`, keys in `.env`. Size: TODO MB (no sampling needed below 200 MB). Checksum after `make pull`: `data/raw.dvc` md5 `TODO`.

**Batch.** `ingest_daily` uses `CronDataIntervalTimetable("@daily", timezone="Europe/Kyiv")`. The batch is the run's logical date (= data interval start) as a Kyiv day, never the current time, so manual runs and backfills pick the same day. The DAG reads files, not Telegram, so it runs without Telegram keys.

**Idempotency.** `load` and `build_features` delete the day, then insert, in one transaction (`warehouse.replace_day`).

| Run of 2026-09-10 | posts_clean | features |
|---|---|---|
| first | TODO | TODO |
| second (`make rerun`) | TODO | TODO |

**Quality gate.** `make bad-batch` feeds `tests/fixtures/bad_batch`: 6 channels whose 60 posts all lost their text (a broken export). `validate` passes (schema is fine), `quality_check` fails: `0 clean rows < 50; 0 channels < 5; empty-text share 1.00 > 0.7`. Thresholds in `configs/quality.yaml`. Screenshots: `docs/screenshots/failed.png`, `docs/screenshots/success.png`.

**Baseline vs model** (test 2026-10-01..27, label = the channel's own ad mark, cut from the text):

| Model | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Hand rule (`rule.py`) | TODO | TODO | TODO | TODO |
| TF-IDF + LR (`train.py`) | TODO | TODO | TODO | TODO |

**Reproduction.** `make reproduce` reads `git_commit` and `data_version` from `dzhynsa-tfidf@champion`, checks the commit out in `.repro/`, runs `dvc pull`, retrains and fails if test PR-AUC differs by more than **0.01** (covers thread and BLAS differences; the split and seeds are fixed).
