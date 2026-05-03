# Stage 1: builder
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# Stage 2: runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

# Install ONLY runtime dependency (curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN groupadd -r appuser && useradd -r -g appuser -u 1000 appuser

COPY data/        ./data/
COPY models/      ./models/
COPY serving/     ./serving/
COPY backtesting/ ./backtesting/
COPY .env         ./.env

RUN mkdir -p mlruns mlartifacts && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]