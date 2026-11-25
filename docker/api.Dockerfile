# Scoring API: the same locked environment as training (uv.lock), so the pickled model
# loads with the scikit-learn version it was trained with. Code is mounted at /opt/project.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 PATH=/opt/venv/bin:$PATH
WORKDIR /opt/project
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project && rm -rf /root/.cache/uv
