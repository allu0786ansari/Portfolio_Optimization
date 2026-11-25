# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN uv venv /opt/venv
ENV PYTHON=/opt/venv/bin/python
COPY pyproject.toml uv.lock README.md ./
COPY scripts/install_deps.sh scripts/
RUN bash scripts/install_deps.sh "dashboard"
COPY src/ src/
RUN uv pip install --python /opt/venv/bin/python --no-deps .

FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -u 1000 -m appuser
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 HOME=/home/appuser
WORKDIR /app
COPY data/__init__.py data/config.py data/
COPY dashboard/ dashboard/
COPY models/__init__.py models/
COPY models/classical/ models/classical/
COPY backtesting/ backtesting/
COPY .streamlit/ .streamlit/
RUN mkdir -p data/raw data/processed && chown -R appuser:appuser /app
USER appuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
