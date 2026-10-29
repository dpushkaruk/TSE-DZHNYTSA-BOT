-include .env
export

setup:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest -q

up:
	docker compose up -d --build

down:
	docker compose down

pull:
	uv run dvc pull

push:
	uv run dvc push

backfill:
	uv run python -m tse_dzhynsa_bot.collect --since 2026-06-01 --until 2026-10-27

collect:
	uv run python -m tse_dzhynsa_bot.collect

marks:
	uv run python -m tse_dzhynsa_bot.count_marks

# make run DAY=2026-09-10: one Kyiv day through the DAG (rerun = clear the same day)
run:
	docker compose exec airflow airflow dags trigger ingest_daily --logical-date $(DAY)T00:00:00+03:00

rerun:
	docker compose exec airflow airflow tasks clear ingest_daily -s $(DAY)T00:00:00+03:00 -e $(DAY)T00:00:00+03:00 -y

# the quality gate on purpose: a day where every post lost its text
bad-batch:
	docker compose exec airflow airflow dags trigger ingest_daily --logical-date 2026-09-11T00:00:00+03:00 --conf '{"raw_dir": "/opt/project/tests/fixtures/bad_batch"}'

count:
	docker compose exec postgres psql -U $${POSTGRES_USER:-dzhynsa} -d warehouse -c "SELECT (SELECT count(*) FROM posts_clean WHERE day='$(DAY)') AS posts_clean, (SELECT count(*) FROM features WHERE day='$(DAY)') AS features"

train:
	uv run python -m tse_dzhynsa_bot.train --register

reproduce:
	uv run python -m tse_dzhynsa_bot.reproduce --alias champion

.PHONY: setup lint test up down pull push backfill collect marks run rerun bad-batch count train reproduce
