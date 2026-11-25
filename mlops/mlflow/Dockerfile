# Pinned MLflow tracking server. Keep MLFLOW_VERSION in step with the client
# version in uv.lock (mlflow-skinny) — v1 installed whatever was latest at start-up.
FROM python:3.12-slim-bookworm
ARG MLFLOW_VERSION=3.18.0
RUN pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" \
    && useradd -r -u 1000 -m mlflow && mkdir -p /mlflow-data && chown mlflow /mlflow-data
USER mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow-data/mlflow.db", \
     "--artifacts-destination", "/mlflow-data/artifacts", "--serve-artifacts", \
     "--allowed-hosts", "*"]
