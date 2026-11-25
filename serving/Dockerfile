# syntax=docker/dockerfile:1.7
# Portfolio API image (Phase 0): locked deps, CPU torch, no secrets baked in.
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN uv venv /opt/venv
ENV PYTHON=/opt/venv/bin/python
COPY pyproject.toml uv.lock README.md ./
COPY scripts/install_deps.sh scripts/
RUN bash scripts/install_deps.sh "serving"
COPY src/ src/
RUN uv pip install --python /opt/venv/bin/python --no-deps .

FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -u 1000 -m appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app
COPY data/__init__.py data/config.py data/
COPY data/ingestion/ data/ingestion/
COPY models/ models/
COPY serving/ serving/
COPY backtesting/__init__.py backtesting/metrics.py backtesting/
RUN mkdir -p data/raw data/processed && chown -R appuser:appuser /app
USER appuser
# Secrets (API_KEY, ...) are injected at run time via env / env_file, never copied into the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1
EXPOSE 8000
CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
